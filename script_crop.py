import cv2

path="/home/pi/vizy/apps/person_detect/output/image_218_3_695_428_vizy_person_event_ 2023-01-17-10-37 .jpg"
img=cv2.imread(path)
l=path.split("_")
print(l)
x=int(path.split("_")[2])
y=int(path.split("_")[3])
w=int(path.split("_")[4])
h=int(path.split("_")[5])
cropped_image = img[y:y+h,x:x+w]
cv2.imshow("Cropped Image",cropped_image)
cv2.imshow("OriginalImage",img)             
#cv2.imwrite("Cropped Image.jpg", cropped_image)
cv2.waitKey(0)
'''
path="/usr/local/lib/python3.7/dist-packages/vizy-0.2.114-py3.7.egg/vizy/media/logo.png"
img=cv2.imread(path)
new_img=cv2.resize(img, (32, 32))
cv2.imwrite("/usr/local/lib/python3.7/dist-packages/vizy-0.2.114-py3.7.egg/vizy/media/logo1.png", new_img)

'''