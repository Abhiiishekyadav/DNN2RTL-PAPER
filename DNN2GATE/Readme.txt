1) Compactness & Robustness-aware Model Selection has been done using Python in the Google-Colab & related (.ipynb) file provided in the model selection folder.
2) Basic building blocks of DNN such as Convolution and Dense layer, ReLu and SoftMax activations, etc. are provided as template functions and given in the template folder.
3) For different applications the mentioned template, the model's structure, and the trained weights aggregate into the high-level description of the model and are provided in the high-level description folder.
4) The obtained high-level description of the fully-trained transform into RTL logic through the HLS process using the Vitis-HLS compiler and the obtained RTL are provided in the RTL folder.
5) The SAT-Robustness-Link file consist of sharpness-aware training code which is utilizes in this work.
   

NOTE: 1) Each (.ipynb) file consists of the results of DPU mapping and generated (.xmodel) file and (.h5) file through the Vitis-ai compiler and the obtained throughput and TOPS unit is mentioned inside the model-selection directory.
2) This repository only consists of RTL implementation of Rock-paper-scissor, Malaria detection, Pneumonia detection, and Gesture recognition models.  To evaluate models for other applications you may follow the given procedure. 


In case of any inconvenience drop a mail to respective email id: pmi2017003@iiita.ac.in, yadav.49@iitj.ac.in
