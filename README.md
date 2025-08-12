# py-imagegrabber

## Installation 

1. Install Python 3.10 
Can be done using Scoop package manager for Windows.

2. Create virtual environment
`python3 -m venv .venv`

3. Activate virtual environment (Powershell)
`.\venv\Scripts\Activate.ps1`

4. Install dependencies
`pip install -r requirements.txt`


## Inroduction
This application takes images using Flir, Lucid Helios and Lucid 2D. The file app.py has considered 1 Flir, 1 Helios, 2 Lucid2D cameras. From the application images get stored in /takenImages/folder. There will be save 6 images from each run: 2D-mono, 2D-rgb, thermo, 3D-ply, heatmap, intensity. The files mainFlir.py, mainHelios.py, mainLucid.py have been used for testing purposes and they show how the classes could be used.  

## Preparations
- In /configs/config.json the correct ip/sn-number to the devices needs to be set.
- run the requirenments.txt to get the correct libaraies installet.

## Run application
to run this application: py app.py

## known bugs
- If the flir-class does not work, try to run testFlir4Histogram.py first. Then try again on the flir-class. 
- the get for camera/stream/color config is not added to the interface class. However thay are implemented in the Helios and Lucid2D class. To see how they work check the files mainHelios.py and mainLucid.py
- the heatmap from the helios camera could be better. 
- The cameraConfig-file for flir should get some more parametes to adjust the apperance.
- the code is quite slow. Dont know why.

## Code structure
The following images shows the class structure.
![Alt text](readmeImg/cameraPackage.png?raw=true "Camera package")
![Alt text](readmeImg/camsyncer.png?raw=true "camsyncer package")
![Alt text](readmeImg/supportiveFeatures.png?raw=true "supportive features package")
![Alt text](readmeImg/classDescription.png?raw=true "detailed class description package")


## Folder structure
```
├───camera/
│   ├───flir/
│   │   └───__init__.py
│   │   └───Flir.py
│   ├───helios/
│   │   └───__init__.py
│   │   └───Helios.py
│   ├───lucid2D/
│   │   └───__init__.py
│   │   └───Lucid2D
│   └───__init__.py
│   └───camera.py
├───camsyncer/
│   └───__init__.py
│   └───camsyncer.py
├───supportiveFeatures/
│   └───__init__.py
│   └───Logger.py
│   └───Timer.py
│   └───readConfig.py
├───configs/
├───takenImages/
├───app.py
├───mainHelios.py
├───mainLucid2D.py
├───mainFlir.py
```
## Cool to know:
To generat class diagrams: pyreverse -o png camera\
it only takes public attributes and methods. And does not give out put.

After running the profiler.py, it was possible to see that the camera-libraries took the most of the time. This explains why the code is so slow. It might not be possible to write the code faster in python. 

# Todo

- [ ] Add smart function that creates takenImages if neccessary