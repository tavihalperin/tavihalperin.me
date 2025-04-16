import os
files = os.listdir ('.')
lines = ["<a href='%s'> %s </a><br/>" %(f,f) for f in files]
with open('alternative_index.html','w') as g:
  print (lines)
  g.write(''.join(lines))


