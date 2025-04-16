import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import imageio, subprocess, os
import torch
# import SyncNetInstance 
import python_speech_features

# from skimage.transform import resize
import pdb
import sys
from skimage.transform import resize
from skimage.color import rgb2gray

flow_types = ['none']
if False:
    try:
        import spynet
        flow_types.append('spynet')
    except: 
        pass
    try:
        import flownet
        flow_types.append('flownet')
    except:
        pass
    try:
        sys.path.append('/cs/labs/peleg/tavih/cs/img/code/vid2speech/pyflow')
        import pyflow
        flow_types.append('pyflow')
    except:
        pass
try:
    import cv2
    flow_types.append('dis')
except:
    pass
flow_type = flow_types[-1]

SR = 16000
FPS = 25
WINDOW_SIZE = 512
OVERLAP = -1 #to be filled from  outside 
WINDOWS_PER_SECOND = -1
WINDOWS_PER_FRAME = -1

batch_size = 32
smooth_div = 20 # TODO change back to 20
dist_thresh = 1000000#.5
relax_end = 1
MAX_WARP = 200

FNULL = open(os.devnull, 'w')

data_dir = '/cs/phd/tavih/tavih_cs_img/code/avsync/data/'
def read_aud(audio_path, verbose=True, save_wav_in_same_dir=False):
    if not audio_path.endswith('.wav'):
        video_path = audio_path
        if not os.path.exists(data_dir + 'my/tmp'):
            os.makedirs(data_dir + 'my/tmp')

        audio_path = data_dir + 'my/tmp/' + os.path.basename(video_path)[:-3] + 'wav'
        audio_path = video_path[:-4] + '.wav'
        command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % 
            (video_path, audio_path)) 
        run_command(command, verbose)    
    
    sr, y = wavfile.read(audio_path)
    assert sr == SR
    return y

def read_vid(video_path):

    reader = imageio.get_reader(video_path)
    # if not reader.get_meta_data()['fps'] == FPS:        
    #     if not os.path.exists(data_dir + 'my/tmp'):
    #         os.makedirs(data_dir + 'my/tmp')
    #     my_video_path = data_dir + 'my/tmp/' + os.path.basename(video_path)
    #     command = ("ffmpeg -y -i %s -vcodec copy -r 25 %s" % 
    #         (video_path, my_video_path))        
    #     run_command(command)
    #     video_path = my_video_path
    #     reader = imageio.get_reader(video_path)

    assert reader.get_meta_data()['fps'] == FPS, 'when changing fps MAKE SURE no new frames are inserted'

    # num_frames = reader.get_length()
    images = []
    try:
        for f in reader:
            images.append(f) 
    except:
        pass

    # if not reader.get_meta_data()['fps'] == FPS:
    #     resample_rate = reader.get_meta_data()['fps'] / FPS
    #     resampled_images = []
    #     for i in range(int(round(len(images) / resample_rate))):
    #         resampled_images.append(images[int(round(resample_rate*i))])
    #     print('resampled video from %.2f fps to 25 fps. # images %d --> %d' %
    #         (reader.get_meta_data()['fps'], len(images), len(resampled_images)))
    #     print(reader.get_meta_data())
    #     images = resampled_images

    print ('number of frames in vid:', len(images))
    return images

def calc_mfcc(aud):
    mfcc = zip(*python_speech_features.mfcc(aud, SR))
    return np.stack([np.array(i) for i in mfcc])

def invert(arr):
    ret = np.zeros(arr.max().astype(np.int))
    for i in range(ret.size):
        m = np.min(np.abs(i-arr))
        ret[i] = np.mean(np.where(np.abs(i-arr) == m)[0])
    return ret

def interpolate_image(vid, f):
    assert f >= -1 and f <= len(vid)
    f = min(max(f,0), len(vid)-1)
    f1 = int(np.floor(f))
    f2 = f1+1
    a1 = f-f1
    a2 = f2-f
    im1 = vid[f1]
    im2 = vid[f2] if f2 < len(vid) else im1

    factor = 1
    
    if flow_type == 'pyflow':    
        factor = 2
        im1 = resize(im1, (im1.shape[0]//factor, im1.shape[1]//factor))
        im2 = resize(im2, (im2.shape[0]//factor, im2.shape[1]//factor))
        im1 = (im1*255).round().astype(np.uint8)
        im2 = (im2*255).round().astype(np.uint8)

    if np.all(im1==im2):
        return im1
    # pdb.set_trace()
    if flow_type == 'dis':
        inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
        inst.setUseSpatialPropagation(True)
        gray1 = (rgb2gray(im1)*255).astype(np.uint8)
        gray2 = (rgb2gray(im2)*255).astype(np.uint8)
        flow12 = inst.calc(gray1, gray2, None)
        flow21 = inst.calc(gray2, gray1, None)
    if flow_type == 'pyflow':
        flow12 = optflow(im1, im2, rgb=True)
        flow21 = optflow(im2, im1, rgb=True)
    elif flow_type == 'spynet':
        flow12 = spynet.optflow(im1, im2)
        flow21 = spynet.optflow(im2, im1)
    elif flow_type == 'flownet':
        flow12 = flownet.optflow(im1, im2)
        flow21 = flownet.optflow(im2, im1)
    elif flow_type == 'none':
        ret = im1 if a1 < a2 else im2
        return ret.astype(np.float32)/255

    warped1 = warp_img(im2, flow12*a2)
    warped2 = warp_img(im1, flow21*a1)
    return warped1*a1 + warped2*a2

def stabilizeSequence(samples, regularization_lambda=30.0):
    length = samples.size
    if length < 6:
        return samples
    
    # find x that solves laplacian_matrix * x = b
    laplacian_matrix = (
        np.diag((1.0 + 2 * regularization_lambda) * np.ones((length - 2,)), 0)
        + np.diag((-1.0 * regularization_lambda) * np.ones((length - 3,)), 1)
        + np.diag((-1.0 * regularization_lambda) * np.ones((length - 3,)), -1))
    
    constraints = np.zeros(length - 2)
    constraints[0] = (regularization_lambda * samples[0])
    constraints[-1] = (regularization_lambda * samples[-1])
    constraints += samples[1:-1]
    
    # inverse_cholesky = np.linalg.inv(np.linalg.cholesky(laplacian_matrix))
    # inverse_laplacian = inverse_cholesky.T @ inverse_cholesky
    # samples = samples.copy()
    # samples[1:-1] = inverse_laplacian @ constraints
    # return samples
    x = samples.copy()
    x[1:-1] = np.linalg.solve(laplacian_matrix, constraints)
    return x

def im2tensor(im):
    im = np.expand_dims(np.stack(im,axis=3), axis=0).transpose((0,3,4,1,2))
    return torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

def aud2tensor(aud):
    mfcc = calc_mfcc(aud)
    cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
    return torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

def run_batch_vid(model, im, ii, lastframe):
    im_batch = [ im[:,:,vframe:vframe+5,:,:] for vframe in range(ii,min(lastframe,ii+batch_size)) ]
    im_in = torch.cat(im_batch,0)
    im_out  = model.__S__.forward_lip(im_in.cuda())
    return im_out.data.cpu()
    
def run_aud_batch(model, cc, ii, lastframe):
    cc_batch = [ cc[:,:,:,vframe*4:vframe*4+20] for vframe in range(ii,min(lastframe,ii+batch_size)) ]
    cc_in = torch.cat(cc_batch,0)
    cc_out  = model.__S__.forward_aud(cc_in.cuda())
    return cc_out.data.cpu()
    
def aud2emb(aud, model):
    if type(aud) == str:
        aud = read_aud(aud)
    lastframe = int(aud.size/SR*FPS) - 6
    y_tensor = aud2tensor(aud)

    cc_feat = torch.cat(
        [run_aud_batch(model, y_tensor, i, lastframe) 
        for i in range(0, lastframe, batch_size)],
        0)
    
    return aud, cc_feat.numpy()

def vid2emb(images, model):
    if type(images) == str:
        images = read_vid(images)
    assert np.all(np.array([224,224,3]) == images[0].shape)
        # images = [resize(im, (224,224)) for im in images]
    lastframe = len(images) - 6
    images_tensor = im2tensor(images)

    im_feat = torch.cat(
        [run_batch_vid(model, images_tensor, i, lastframe) 
        for i in range(0,lastframe,batch_size)], 
        0)

    return images, im_feat.numpy()

def resample_vid(vid, prefix=''):
    if not os.path.exists(data_dir + 'my/tmp/'):
        os.makedirs(data_dir + 'my/tmp/')
    target = data_dir + 'my/tmp/' + prefix+ os.path.basename(vid)
    command = ("ffmpeg -y -i %s -qscale:v 4 -async 1 -r 25 -deinterlace %s" % (vid, target))
    run_command(command)
    return target

def run_command(command, verbose=True):
    print('runing: ', command if verbose else '...' , end=' ')
    ret =  subprocess.call(command, shell=True, stdout=FNULL, stderr=FNULL)
    if ret :
        print('\nFailed!!!')
        sys.exit()
    else:
        print('.... OK!')

def aud2spec(aud):
    return signal.stft(aud, nperseg=WINDOW_SIZE, noverlap=OVERLAP)[2]

def spec2aud(spec):
    return signal.istft(spec, nperseg=WINDOW_SIZE, noverlap=OVERLAP)[1]

def make_monotonic(flow):
    flow = np.maximum.accumulate(flow + np.arange(flow.shape[0]))
    return flow - np.arange(flow.shape[0])

def warp_img(img, flow):
    if img.dtype == np.uint8:
        img = img.astype(np.float32)/255
    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    coordiantes = [yy+flow[...,0], xx+flow[...,1]]
    return np.dstack([map_coordinates(img[...,i], coordiantes, mode='reflect')
        for i in range(3)])

def warp_spec(spec, warp):
    xx, yy = np.meshgrid(np.arange(warp.size), np.arange(spec.shape[0]))
    coordiantes = [yy, warp+xx]
    warped_real = map_coordinates(spec.real, coordiantes, mode='reflect', order=1)
    warped_imag = map_coordinates(spec.imag, coordiantes, mode='reflect', order=1)
    return warped_real + 1j*warped_imag

def data_dist(d1,d2):
    dist = (np.mean((d1-d2)**2))
    return np.minimum(dist,dist_thresh)

def cos(d1,d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    return (d1 @ d2)

def resize_signal(signal, new_length):
    new_length = int(new_length)
    signal = signal * new_length / signal.size
    coordinates = np.linspace(0, signal.size-1, new_length)
    return map_coordinates(signal, [coordinates], order=1)


def minimal_dist(l1, l2, i, j, dist_function):
    return np.min([[dist_function(l1[channel1][i],l2[channel2][j]) 
        for channel1 in range(len(l1))] for channel2 in range(len(l2))])


def dp_new(l1, l2, max_warp=MAX_WARP, dist_function=data_dist, prefer_delay=False):
    if not type(l1) == list:
        l1 = [l1]
        l2 = [l2]
    mat = np.ones((l1[0].shape[0]+1,l2[0].shape[0]+1))*np.inf
    mat[0,:relax_end] = 0
    mat[:relax_end,0] = 0
    orig = np.zeros((l1[0].shape[0],l2[0].shape[0], 2), dtype=int)
    for i in range(mat.shape[0]-1):
        # for j in range(mat.shape[1]-1):
        diag_i = i*mat.shape[1]//mat.shape[0]
        for j in range(max(0, diag_i-max_warp), min(diag_i+max_warp, mat.shape[1]-1)):
            if prefer_delay:
                data_costs =( .5*minimal_dist(l1, l2, i, j, dist_function) +
                              .25*minimal_dist(l1, l2, max(i-1,0), j, dist_function) +
                              .25*minimal_dist(l1, l2, max(i-2,0), j, dist_function))
            else:
                data_costs = minimal_dist(l1, l2, i, j, dist_function)
            smooth_costs = np.array([1,0,1])/smooth_div
            cost = data_costs + smooth_costs + [mat[i, j+1], mat[i,j], mat[i+1, j]]
            argmin = np.argmin(cost)
            orig[i,j] = [[i-1,j], [i-1,j-1], [i,j-1]][argmin]
            mat[i+1,j+1] = np.min(cost)
    return mat[1:,1:], orig

def find_path_new(dp_mat, dp_orig):
    amin1 = np.argmin(dp_mat[-1, -relax_end:]) + dp_mat.shape[1]-relax_end
    amin2 = np.argmin(dp_mat[-relax_end:, -1]) + dp_mat.shape[0]-relax_end
    if dp_mat[-1, amin1] < dp_mat[amin2, -1]:
        i,j = dp_orig[-1, amin1]
        cost = dp_mat[-1, amin1]
        path = [[dp_orig.shape[0]-1, amin1]]
    else:
        i,j = dp_orig[amin2, -1]
        cost = dp_mat[amin2, -1]
        path = [[amin2, dp_orig.shape[1]-1]]
    while i>=0 and j>=0:
        path.append((int(i), int(j)))
        i,j = dp_orig[i,j]
    path = path[::-1]
    return np.array(path), cost

def griffin_lim(spec, iterations=100):
    mag = np.abs(spec)
    # spec = mag * np.exp(1j*np.random.uniform(low=-np.pi, high=np.pi, size=spec.shape))
    for i in range(iterations):
        wav = signal.istft(spec, nperseg=WINDOW_SIZE, noverlap=OVERLAP)[1]
        phase = np.angle(signal.stft(wav, nperseg=WINDOW_SIZE, noverlap=OVERLAP)[2])
        spec = mag * np.exp(1j* phase)
    return wav

def phase_vocoder(spec, time_steps):
    hop_length=WINDOW_SIZE-OVERLAP

    xx, yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))
    xx = np.zeros_like(xx) 
    coordiantes = [yy, time_steps+xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, spec.shape[0])

    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):

        # Store to output array
        warped_spec[:, t] *= np.exp(1.j * phase_acc)

        # Compute phase advance
        # print (t, step, time_steps.shape, spec_angle.shape)     
        # print(time_steps) 
        dphase = (spec_angle[:, step+1] - spec_angle[:, step] - phi_advance)

        # Wrap to -pi:pi range
        dphase = np.mod(dphase-np.pi, 2*np.pi)-np.pi #dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return warped_spec

def optflow(xx1, xx2, rgb=True):
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = [1,0][rgb]  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    if colType:
        im1 = xx1[...,None].astype(np.float64).copy()
        im2 = xx2[...,None].astype(np.float64).copy()
    else:
        im1 = xx1.astype(np.float64)/255
        im2 = xx2.astype(np.float64)/255
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    return np.dstack([u,v])
