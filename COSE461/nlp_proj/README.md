# nlp_proj

COSE461 semester project

Sarcasm detection
- Aims to improve the sarcasm detection model
- Add a response generation
- Possibly detect sarcasm between languages?
    For example, they say "Would you like a cup of tea?" in Japan often refers to sarcasm (Or a lie, to be more accurate)
    The same example in Seoul does not directly connect to sarcasm --> The context of each countries' culture would also matter

_____________________________________________________________________________________________________

Specifications:

Windows 11 64-bit
Ryzen R9-8845hs
NVIDIA RTX 4060 Laptop GPU
32GB LPDDR5X-6400MHz
_____________________________________________________________________________________________________

Python 3.10.0rc2 64-bit
_____________________________________________________________________________________________________

TensorFlow version: 2.10.0
_____________________________________________________________________________________________________

NVIDIA CUDA 11.8

$ nvidia-smi
Wed May  8 17:47:45 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 552.22                 Driver Version: 552.22         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   47C    P8              3W /   86W |    1300MiB /   8188MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2460    C+G   ...s\System32\SystemSettingsBroker.exe      N/A      |
|    0   N/A  N/A      3636    C+G   ...m\radeonsoftware\RadeonSoftware.exe      N/A      |
|    0   N/A  N/A      6568    C+G   C:\Windows\explorer.exe                     N/A      |
|    0   N/A  N/A      7072    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A      7336    C+G   ...les\Microsoft OneDrive\OneDrive.exe      N/A      |
|    0   N/A  N/A      9208    C+G   ...nr4m\radeonsoftware\AMDRSSrcExt.exe      N/A      |
|    0   N/A  N/A     10632    C+G   ...nt.CBS_cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A     10656    C+G   ...2txyewy\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A     11368    C+G   ...on\124.0.2478.80\msedgewebview2.exe      N/A      |
|    0   N/A  N/A     12956    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A     12976    C+G   ...oogle\Chrome\Application\chrome.exe      N/A      |
|    0   N/A  N/A     13124    C+G   ...t.LockApp_cw5n1h2txyewy\LockApp.exe      N/A      |
|    0   N/A  N/A     14928    C+G   ...crosoft\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A     16428    C+G   ...B\system_tray\lghub_system_tray.exe      N/A      |
|    0   N/A  N/A     16452    C+G   ...5n1h2txyewy\ShellExperienceHost.exe      N/A      |
|    0   N/A  N/A     17308    C+G   ...\Local\slack\app-4.38.115\slack.exe      N/A      |
|    0   N/A  N/A     17956    C+G   ...on\124.0.2478.80\msedgewebview2.exe      N/A      |
|    0   N/A  N/A     19236    C+G   ...ograms\cron-web\Notion Calendar.exe      N/A      |
|    0   N/A  N/A     20488    C+G   ...Programs\Microsoft VS Code\Code.exe      N/A      |
|    0   N/A  N/A     22084    C+G   ...BridgeWPF\SamsungNotesBridgeWPF.exe      N/A      |
|    0   N/A  N/A     24180    C+G   ...k6g\QuickControls\QuickControls.exe      N/A      |
|    0   N/A  N/A     24300    C+G   ...siveControlPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A     24780    C+G   ..._x64__3c1yjt4zspk6g\BudsManager.exe      N/A      |
|    0   N/A  N/A     26348    C+G   ...\cef\cef.win7x64\steamwebhelper.exe      N/A      |
|    0   N/A  N/A     27716    C+G   ...Google\NearbyShare\nearby_share.exe      N/A      |
+-----------------------------------------------------------------------------------------+
_____________________________________________________________________________________________________

cuDNN 8.9.1

$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
_____________________________________________________________________________________________________

Environmental variables
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
_____________________________________________________________________________________________________

Installed Microsoft C++ Build tools 14.0 or greater
_____________________________________________________________________________________________________

Trained on:
- 1.3 million reddit comments *One thing to note is that for reddit comments specifically, context is given as well (mother comment). Hence, I would train in the order of huggingface -> twitter(kaggle) -> reddit comments with context(kaggle) to see the improving accuracy

- context가 있는 데이터셋을 훈련하는데 eval_dataset 사용하여 overfitting 방지 / context_0 파일 학습하니 오히려 정확도가 떨어지는 현상 발생 (testing 데이터의 한계일수도? 트위터로만 테스트 하는거니까)

- context_1.csv 파일부턴 max_len 256으로 수정하여 학습 시작 (시간은 훨씬 더 오래걸릴듯) / nvm changed back to 128 for consistency --> had the exact same accuracy

- context_2.csv 파일부턴 underfitting 방지를 위한 epoch 3->5로 증가, earlystop 사용해서 overfitting 방지
결국 같은 학습, 6%에서 학습 중단 (probably due to earlystop?)

아마 레딧의 데이터들이 flawed 되어있는 가능성이 높아보임

changed my idea, planning on giving two inputs at the same time (one image, one text)
the image will serve as a context 