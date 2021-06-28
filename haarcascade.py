import cv2  
#OpenCv kütüphanesini dahil ediyoruz
import numpy as np 
#Numpy kütüphanesini dahil ediyoruz

img= cv2.imread('yuz.jpg') 
#yuz.jpg yazan yere dosyanız içerisine atmış olduğunuz görüntülerin adını yazınız.

faceCasc =cv2.CascadeClassifier('haarcascade_frontalface_default.xml ')
#İndirmiş olduğumuz haar-cascade sınıfını dahil ediyoruz.Adını değiştirebilirsiniz.

grayColors=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
#Görüntümüzü gri yapıyoruz.Daha doğru sonuçlar almak için

faces= faceCasc.detectMultiScale(grayColors,1.1,3)
#Görüntüyü 1,1 skala ile kontrol edecek ve 3 kere yüzün orada var mı yok mu olduğunu teyit edecek.


#Görüntüde bulunan yüzlere dikdörtgen çizmek için ağağıdaki döngüyü kullanıyoruz. 
#For ile dönmemizin nedeni ise görüntü içerisinde birden fazla insan olabilir. 
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #Çerçevenin konumu, rengi ve kalınlığı

cv2.imshow('faces',img) #Görüntüyü göster.
cv2.waitKey(0)
cv2.destroyAllWindows()
