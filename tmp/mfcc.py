import time, os, argparse, pdb
start = time.time()

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import imageio
import numpy as np
import torch
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
# import SyncNetInstance 
from scipy.ndimage.interpolation import map_coordinates
import avsnap_utils as utils
import warnings

# warnings.filterwarnings('error')


# 1) python3 run_pipeline.py --videofile=.mp4
# 2) use the ouptputed video from data/pycrop/...

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=512, 
    help='number of samples audio window')
parser.add_argument('--audio_overlap', type=int, default=256, 
    help='number of samples for overlap between successive audio windows (out of WINDOW_SIZE)')
# parser.add_argument('--modality', type=str, default='aa', 
#     help='aa (audio vs. audio) | vv (video vs. video) | av | va')
parser.add_argument('--griffin_lim_iterations', type=int, default=5, help='')

parser.add_argument('--out_prefix', type=str, default=None, 
    help='prefix where to save output')

parser.add_argument('reference', type=str)
parser.add_argument('unaligned', type=str)

opt = parser.parse_args()

# model = SyncNetInstance.SyncNetInstance()

# model.loadParameters(utils.data_dir + "syncnetl2.model")#opt.initial_model)
# print("Model loaded.")#%opt.initial_model);

# model.__S__.eval()

utils.WINDOW_SIZE = opt.window_size
utils.OVERLAP = opt.audio_overlap #1,2,4,5,8,10,16,20,32,40,64,80,128,160,320,
utils.WINDOWS_PER_SECOND = utils.SR/(utils.WINDOW_SIZE-utils.OVERLAP)
utils.WINDOWS_PER_FRAME = utils.WINDOWS_PER_SECOND/utils.FPS

# max_diff_sync = opt.max_diff_smooth 
# gaussian_sigma = opt.gaussian_sigma 

class Data:
    def __init__(self):

        self.reference = utils.resample_vid(opt.reference, 'reference_')
        self.unaligned = utils.resample_vid(opt.unaligned, 'unaligned_')

        self.vid1 = utils.read_vid(self.reference)
        self.vid2 = utils.read_vid(self.unaligned)

        self.aud1 = utils.read_aud(self.reference)
        self.aud2 = utils.read_aud(self.unaligned)

        # self.emb1 = utils.aud2emb(self.aud1, model)[1]
        # self.emb2 = utils.aud2emb(self.aud2, model)[1]

        self.mfcc1 = utils.calc_mfcc(self.aud1).T [2*4:-3*4] # to be comparable to embeddings which are only computed for frames (centers) 2....-4
        self.mfcc2 = utils.calc_mfcc(self.aud2).T [2*4:-3*4] # to be comparable to embeddings which are only computed for frames (centers) 2....-4

#        print(self.mfcc1.shape, self.mfcc2.shape)
        self.spec1 = utils.aud2spec(self.aud1)
        self.spec2 = utils.aud2spec(self.aud2)


def warp_vid_towards_aud(spec, vid, path, out_folder):
    if path.max() >= spec.shape[1]:
        print('flow field reaches outside source for warp')
        # if path.max() - spec.shape[1] > utils.WINDOWS_PER_FRAME:
        #     pdb.set_trace()
        spec = np.pad(spec, ((0,0), (0, int(path.max() - spec.shape[1]+1))), mode='constant')

    warped_complex_dp = utils.phase_vocoder(spec, path)
    aud_dp = utils.spec2aud(warped_complex_dp)

    print('applying griffin lim')
    start_gl = time.time()
    aud_dp_gl = utils.griffin_lim(warped_complex_dp, opt.griffin_lim_iterations)
    print('griffin lim on one aud took: ', time.time()-start_gl)

    warped_aud_file_gl = utils.data_dir + 'tmp_dp_gl.wav'
    wavfile.write(warped_aud_file_gl, rate=utils.SR, data=aud_dp_gl/aud_dp_gl.max())

    #create baseline streched audio
    linear_path = np.linspace(0, spec.shape[1]-1, path.size)
    streched_spec = utils.phase_vocoder(spec, linear_path)
    start_gl = time.time()
    aud_streched = utils.griffin_lim(streched_spec, opt.griffin_lim_iterations)

    baseline= utils.data_dir + 'tmp_streched.wav'
    wavfile.write(baseline, rate=utils.SR, data=aud_streched/aud_streched.max())
    wavfile.write(warped_aud_file_gl, rate=utils.SR, data=aud_dp_gl/aud_dp_gl.max())


    original_vid_clip_file = utils.data_dir + 'tmp_original_vid.avi'
    original_vid_clip = imageio.get_writer(original_vid_clip_file, fps=utils.FPS)
    for f in vid:
        original_vid_clip.append_data(f)
    original_vid_clip.close()
    command = (
        'ffmpeg -y -i %s -i %s -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest %s' % 
        (original_vid_clip_file, warped_aud_file_gl, out_folder + 'warped.avi'  ))
    utils.run_command(command, False)
    command = (
        'ffmpeg -y -i %s -i %s -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest %s' % 
        (original_vid_clip_file, baseline, out_folder + 'baseline.avi'  ))
    utils.run_command(command, False)



def smooth_flow(path, prefix, suffix):
    factor = utils.WINDOWS_PER_SECOND
    max_diff_sync = .2
    gaussian_sigma = 50
    path = utils.resize_signal(path - np.arange(path.size), 
        utils.WINDOWS_PER_FRAME * path.size)

    max_diff = 0
    sigma = 1
    print ('smooth flow field')
    while max_diff < max_diff_sync and sigma < 10000:
        sigma *= 1.5 
        path_smoothed = utils.stabilizeSequence(path, sigma)
        max_diff = np.abs(path_smoothed/factor - path/factor).max()
        print ('s',sigma, max_diff)
    print('max smoothness (seconds): ', max_diff)
    print('sigma', sigma)

    path_smoothed = gaussian_filter(path_smoothed, gaussian_sigma)
    print('max path diff in seconds (after smoothing): ', 
        np.abs(path/factor - path_smoothed/factor).max())
    print('(found at)): ', 
        np.argmax(np.abs(path/factor - path_smoothed/factor)))

    plt.figure()
    plt.plot(path / factor)
    plt.plot(path_smoothed / factor)
    plt.savefig(prefix + '_dp_smooth_%s.png' % suffix, dpi=150)
    print(prefix + '_dp_smooth_%s.png' % suffix)
    return path_smoothed + np.arange(path_smoothed.size)


def main():
    data = Data()

    # pdb.set_trace()
    # make sure no suprises from ffmpeg
    print ('mfcc1:', data.mfcc2.shape[0], 'frames1*4:', 4*(len(data.vid2)-6))
    print ('mfcc2:', data.mfcc1.shape[0], 'frames2*4:', 4*(len(data.vid1)-6))
    print ('spec2 windows/win_per_frame:', data.spec2.shape[1]/utils.WINDOWS_PER_FRAME , 'frames2:',len(data.vid2))
    print ('aud1 size/sr*fps', data.aud1.size/utils.SR*utils.FPS , 'frames1:',len(data.vid1))

    if opt.out_prefix is None:
        vid1_base_name = os.path.basename(data.reference)[:-4]
        vid2_base_name = os.path.basename(data.unaligned)[:-4]
        opt.out_prefix = utils.data_dir + 'gt_%s_%s_' % (vid1_base_name, vid2_base_name)
    out_folder = opt.out_prefix

    if not os.path.exists(os.path.dirname(out_folder)):
        os.makedirs(os.path.dirname(out_folder))
    
    print('out will be saved to ', out_folder)

    def l2_dist(v1, v2):
        return np.linalg.norm(v1-v2)

    dp_mat, dp_orig = utils.dp_new(data.mfcc1, data.mfcc2, 200, l2_dist)
    path, cost = utils.find_path_new(dp_mat, dp_orig)
    print ('dynamic programming cost', cost)

    # adding 2 because the embedding is generated from 5 frames and refers to the middle one
    lowest_start1 = path[0,0] +2
    lowest_start2 = path[0,1] +2
    lowest_end1 = lowest_start1+path[-1,0]*4+1 #multiply by 4 because there are 4 mfcc coeff per video frame
    lowest_end2 = lowest_start2+path[-1,1]*4+1

    path1_2 = np.array([path[i,1] for i in range(1,path.shape[0]) 
        if path[i,0] > path[i-1,0]])
    path1_2 = np.r_[path[0,1], path1_2]
    path2_1 = np.array([path[i,0] for i in range(1,path.shape[0]) 
        if path[i,1] > path[i-1,1]])
    path2_1 = np.r_[path[0,0], path2_1]

    data.aud1 = data.aud1[int(lowest_start1/utils.FPS*utils.SR):
                          int(lowest_end1  /utils.FPS*utils.SR)]
    data.aud2 = data.aud2[int(lowest_start2/utils.FPS*utils.SR):
                          int(lowest_end2  /utils.FPS*utils.SR)]
    data.spec1 = data.spec1[:, int(lowest_start1 * utils.WINDOWS_PER_FRAME):
                               int(lowest_end1   * utils.WINDOWS_PER_FRAME)]
    data.spec2 = data.spec2[:, int(lowest_start2 * utils.WINDOWS_PER_FRAME):
                               int(lowest_end2   * utils.WINDOWS_PER_FRAME)]
    data.vid1 = data.vid1[lowest_start1:lowest_end1]
    data.vid2 = data.vid2[lowest_start2:lowest_end2]

    dists = np.linalg.norm(data.mfcc2[None] - data.mfcc1[:,None], axis=(2))
    dists_im = dists[lowest_start1:lowest_end1, lowest_start2:lowest_end2]
    dists_im = dists_im/dists_im.max()
    dists_im = np.dstack([dists_im]*3)

    plt.figure()
    plt.imshow(dp_mat)
    plt.plot(path2_1,'r')
    plt.savefig(out_folder + '_dp_comulative.png', dpi=150)

    plt.figure()
    plt.imshow(dists_im)
    plt.plot(path2_1,'r')
    plt.savefig(out_folder + '_dp.png', dpi=150)
    

    path = path1_2 
    # path = smooth_flow(path, out_folder, 'tmp') # TODO remove this line
    path = utils.resize_signal(path , data.spec1.shape[1])
    # path = utils.resize_signal(path , utils.WINDOWS_PER_FRAME * path.size)
    np.savez(out_folder + 'flow', path=path, path_inverse=utils.invert(path))

    # *audio* 2 should be warped to match *video* 1
    warp_vid_towards_aud(data.spec2, data.vid1, path, out_folder)

if __name__ == '__main__':
    main()

print('total time:', time.time()-start)
