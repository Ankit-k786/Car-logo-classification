==================================================<br />
[Install Dependencies]:

	1. Install python>=3.8 if not already install.
	2. Install other dependency usin "pip install -r requirements.txt"
=================================================<br />
[Run Inference on single image]:

	1. Set the path of the logo Image and pretrained model in inference.py file.
	2. Run the python inference.py file.
	3. It will predict the class of image logo.
=================================================<br />
[Run Test.py for genrate Confusion matrix and accuracy]:
	
	1. Set the path of the val folder and pretrained model in test.py file.
	2. Run the python test.py file.
	3. It will genrate the confusion matrix and save as confusion matrix.png, It will also calculate the accuracy 
	   and precision,recall,fscore for each indivisual class.

==================================================<br />
[Run Train.py for traing]:
	1. Set the path of the train folder in train.py file.
	2. Run the python train.py file.
	3. It will save the trained weight in ./Car_logo_classification/model/ name as "logo_classify_weight.h5"<br />
===============================================<br />
Note:Unzip the data.zip . Training and test data is in data folder<br />

[Check and regenrate the results]:

	1. I trained the classifier model for logo classification having 28 categories and model input_size=(50,50,3).
	2. I got 87% Accuracy and the weight file is present name as "./Car_logo_classification/logo_classify_weight.h5".
	3. You can see the confusion matrix which is saved name as "./Car_logo_classification/confusion matrix.png".
	4. Regenrate the result just run test.py python file.
	5. By running the test.py you can check accuracy and (precision,recall,fscore for each logo class).
	
	6. logo_classes names=['Fiat.common', 'Kia.new', 'unknown', 'Renault.common', 'Tata.text', 'Volkswagen.common', 'Skoda.common', 
         'Nissan.common', 'Jeep.common', 'Mercedes-Benz.common', 'Hyundai.common', 'Toyota.common','Jaguar.frontal',
         'Honda.common', 'Kia.common', 'MG-Motor.common', 'ISUZU.common', 'Volvo.frontal', 'Datsun.common', 
         'Mahindra.common', 'Ford.common', 'Jaguar.rear', 'Maruti-Suzuki.common', 'BMW.common', 'Chevrolet.common',
         'Mitsubishi.common', 'Tata.common', 'Audi.common']
	 
Confusion Matrix: 
<img src="https://github.com/Ankit-k786/Car-logo-classification/blob/main/confusion_matrix.png" width="512"/>

**Note: Download Pretrained weight from this drive link:**
https://drive.google.com/file/d/1VyILzc1zE4AaOhWdoFoZd2Yu4tvAW6f0/view?usp=share_link

                       
                     
                
