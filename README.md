# nanobot - Autonomous on Playing and interacting with game from screenshot only

```
git clone https://github.com/diopthe20/nanobot
git submodule update --init --recursive

```

### TODO

- Develop a deployment strategy

### Vision model
We will use some vision model to get the information about what we will see in the screen

| name                                                                          | status | description                                                                                                                                                       |
| ----------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [llama-3-vision-alpha](https://huggingface.co/qresearch/llama-3-vision-alpha) |        | projection module trained to add vision capabilties to Llama 3 using SigLIP. built by [@yeswondwerr](https://x.com/yeswondwerr) and [@qtnx_](https://x.com/qtnx_) |

### OCR 

I used EasyOCR for recognize some text in the screen during gathering. 
You can go to https://huggingface.co/spaces/tomofi/EasyOCR to test with EasyOCR
### Event Handling

We take the environment state as an event and send it to event handler


## Object Detection



Label with Label Studio, Export to YOLO Format and then Upload to ROBOFLOW to export to the right format for YOLO

Train with YOLO

=> Predict from the screen stream

Currently this project in development. The current phase is try out new probilities
