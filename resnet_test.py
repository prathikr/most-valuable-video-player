import torch
import numpy as np
import librosa
import pickle
import os
from pydub import AudioSegment
from pydub.utils import make_chunks
import re
import torch.nn.functional as nnf
import matplotlib.pyplot as plt

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db

def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

print("setting hardware backend...")
if torch.cuda.is_available():
  device=torch.device('cuda:0')
elif torch.backends.mps.is_available():
  device=torch.device('mps')
else:
  device=torch.device('cpu')
print(f"{device} backend set")

resnet_model = torch.load('reset_model.pt')
f = open("categories.pkl","rb")
index2category = pickle.load(f)
f.close()

filename='audio/1-1-A-51.wav'
spec=spec_to_image(get_melspectrogram_db(filename))
spec_t=torch.tensor(spec).to(device, dtype=torch.float32)
pr=resnet_model.forward(spec_t.reshape(1,1,*spec_t.shape))
ind = pr.argmax(dim=1).cpu().detach().numpy().ravel()[0]
print(index2category[ind])

filename='audio/5-263831-B-6.wav'
spec=spec_to_image(get_melspectrogram_db(filename))
spec_t=torch.tensor(spec).to(device, dtype=torch.float32)
pr=resnet_model.forward(spec_t.reshape(1,1,*spec_t.shape))
ind = pr.argmax(dim=1).cpu().detach().numpy().ravel()[0]
print(index2category[ind])

myaudio = AudioSegment.from_file("los-altos-comeback.wav" , "wav") 
chunk_length_ms = 1000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
four_sec_silence = AudioSegment.silent(duration=4000)

#Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
    chunk_name = "tmp/chunk{0}.wav".format(i)
    chunk += four_sec_silence
    chunk.export(chunk_name, format="wav")

mypath = "tmp/"
onlyfiles = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(mypath) for f in filenames]
onlyfiles.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0]) # sort by chunk number

y = []

for f in onlyfiles:
    spec=spec_to_image(get_melspectrogram_db(f))
    spec_t=torch.tensor(spec).to(device, dtype=torch.float32)
    pr=resnet_model.forward(spec_t.reshape(1,1,*spec_t.shape))
    prob = nnf.softmax(pr, dim=1)

    top_p, top_class = prob.topk(5, dim = 1)
    keys = [index2category[ind] for ind in top_class.cpu().detach().numpy().ravel()]
    values = top_p.cpu().detach().numpy().ravel()
    res = dict(zip(keys, values))
    print(f, str(res))
    y.append(res.get('whistle', 0))

y_clip = np.clip(y, a_min = 0.75, a_max = 1)
plt.plot(range(len(onlyfiles)), y_clip, 'ro')
plt.show()

    # ind = pr.argmax(dim=1).cpu().detach().numpy().ravel()[0]
    # print(f, index2category[ind])