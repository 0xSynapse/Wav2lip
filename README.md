# Wav2Lip Lip-Syncing Model

This document provides an overview of the Wav2Lip lip-syncing model implementation using Python in Google Colaboratory. This code allows you to synchronize lip movements in a video with an audio track.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xC6fub2FsFl2SGUaOnXoAvhqyrW_vOjA?usp=sharing)

## How to Run the Code

To run the code, follow these steps:

1. **Set Up Wav2Lip**: This step installs dependencies and downloads pre-trained models. Run the following code cell to set up the environment:

    ```python
    # Run this code cell to set up Wav2Lip
    !rm -rf /content/sample_data
    !mkdir /content/sample_data

    !git clone https://github.com/justinjohn0306/Wav2Lip

    # Install all the requirements
    !cd Wav2Lip && pip install -r requirements_colab.txt

    #download the pretrained model
    !wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth' -O 'checkpoints/wav2lip.pth'
    !wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth' -O 'checkpoints/wav2lip_gan.pth'
    !wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/resnet50.pth' -O 'checkpoints/resnet50.pth'
    !wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/mobilenet.pth' -O 'checkpoints/mobilenet.pth'
    #download pretrained model for face detection
    !wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"
    a = !pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl
    !pip install git+https://github.com/elliottzheng/batch-face.git@master

    !pip install ffmpeg-python mediapipe==0.8.11
    !pip install librosa==0.9.1

    #importing libraries

    from IPython.display import HTML, Audio
    from google.colab.output import eval_js
    from base64 import b64decode
    import numpy as np
    from scipy.io.wavfile import read as wav_read
    import io
    import ffmpeg
    from ghc.l_ghc_cf import l_ghc_cf

    from IPython.display import clear_output
    clear_output()
    print("\nDone")
    ```
   
2. **Select Video**: In this step, you can upload a video from your local drive. Ensure that the video duration does not exceed 60 seconds. The code will automatically resize the video to 720p if needed.

3. **Select Audio**: You can  upload an audio file from your local drive.

4. **Run Lip-Syncing**: This section performs lip-syncing on the selected video and audio. It provides two options: one with smoothing and one without smoothing and few other features too. Run either of the following code cells to start the lip-syncing process:
   
   ```python
    %cd /content/Wav2Lip
    # Set up paths and variables for the output file
    output_file_path = '/content/Wav2Lip/results/result_voice.mp4'

    # Delete existing output file before processing, if any
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    pad_top =  0#@param {type:"integer"}
    pad_bottom =  10#@param {type:"integer"}
    pad_left =  0#@param {type:"integer"}
    pad_right =  0#@param {type:"integer"}
    rescaleFactor =  2#@param {type:"integer"}
    #Prevent smoothing face detections over a short temporal window
    nosmooth = False #@param {type:"boolean"}
   ```

    - Lip-Syncing Using wav2lip:
    
        ```python
        checkpoint_path = 'checkpoints/wav2lip.pth'

        if nosmooth == False:
          !python inference.py --checkpoint_path $checkpoint_path --face "../input_video.mp4" --audio "../input_audio.wav" --pads $pad_top $pad_bottom $pad_left $pad_right --resize_factor $rescaleFactor
        else:
          !python inference.py --checkpoint_path $checkpoint_path --face "../input_video.mp4" --audio "../input_audio.wav" --pads $pad_top $pad_bottom $pad_left $pad_right --resize_factor $rescaleFactor --nosmooth

        #Preview output video
        if os.path.exists(output_file_path):
            clear_output()
            print("Final Video Preview")
            print("Download this video from", output_file_path)
            showVideo(output_file_path)
        else:
            print("Processing failed. Output video not found.")
        ```

    - Lip-Syncing Using wav2lip_gan:
    
       ```python
        checkpoint_path = 'checkpoints/wav2lip_gan.pth'

        if nosmooth == False:
          !python inference.py --checkpoint_path $checkpoint_path --face "../input_video.mp4" --audio "../input_audio.wav" --pads $pad_top $pad_bottom $pad_left $pad_right --resize_factor $rescaleFactor
        else:
          !python inference.py --checkpoint_path $checkpoint_path --face "../input_video.mp4" --audio "../input_audio.wav" --pads $pad_top $pad_bottom $pad_left $pad_right --resize_factor $rescaleFactor --nosmooth

        #Preview output video
        if os.path.exists(output_file_path):
            clear_output()
            print("Final Video Preview")
            print("Download this video from", output_file_path)
            showVideo(output_file_path)
        else:
            print("Processing failed. Output video not found.")
        ```

5. **Evaluate Performance**: After running the lip-syncing process, you can evaluate the performance by previewing the output video. The code will display the final lip-synced video in the Colab notebook, and you can download it for further inspection.
   
   ![Example Videos](https://drive.google.com/drive/folders/1P_TUmE-1SV8uSMVaHQQ89Ue-Nx4z46zN?usp=sharing)

## List of Third-Party Libraries

The code makes use of the following third-party libraries:

- [moviepy](https://zulko.github.io/moviepy/): A Python library for video editing and manipulation.
- [OpenCV](https://opencv.org/): An open-source computer vision and machine learning software library.
- [numpy](https://numpy.org/): A fundamental package for scientific computing with Python.
- [scipy](https://www.scipy.org/): A library for mathematics, science, and engineering.
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python): A Pythonic interface for FFmpeg.
- [mediapipe](https://mediapipe.dev/): A framework for building multimodal (audio, video, etc.) applied machine learning pipelines.
- [librosa](https://librosa.org/doc/main/index.html): A Python package for analyzing and manipulating audio data.

Please ensure you have these libraries installed to run the code successfully.

For more details on each step and additional options, refer to the code comments and Colab notebook cells.

**Note**: Ensure you have a Google Colab account and access to a GPU for faster processing when running this code.

