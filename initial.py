from flask import Flask, request, jsonify
import werkzeug
import pytesseract
import cv2
app = Flask(__name__)

@app.route('/upload', methods=["GET","POST"])
def upload():
    global response 
    if request.method == "POST" :
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("\nReceived image File name : " + imagefile.filename)
        imagefile.save("./uploadedimages/" + "image.png")
        image = cv2.imread("uploadedimages/image.png")
        base_image = image.copy()
        image_preprocess = image_into_text(image)
        no_noise = noise_removal(image_preprocess)
        cv2.imwrite("temp/notaaasss_noise.jpg", no_noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,25))
        dilate = cv2.dilate(no_noise, kernel, iterations=1)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
        main_text = ""
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if h > 200 and w > 250:
                roi = image_preprocess[y:y+h, 0:x]
#         cv2.rectangle(image, (0, y), (x, 0 + h+20), (36,255,12), 2)
        
                constant= cv2.copyMakeBorder(roi.copy(),30,30,30,30,cv2.BORDER_CONSTANT,value=[255,255,255])
                ocr_result = pytesseract.image_to_string(constant)
        #cv2.imwrite("temp/outputs.png", roi)
                print (ocr_result)
        ocr_result = pytesseract.image_to_string(image_preprocess)
        print (ocr_result)
        return jsonify({
            "message": ocr_result,
        })
def image_into_text(image):
  base_image= image.copy()
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (7,7), 0)
  #thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  thresh, im_bw = cv2.threshold(gray, 120, 200, cv2.THRESH_BINARY)
  
  return(im_bw) 
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 1)
    return (image)


if __name__ == "__main__":
    app.run(debug=True, port=4000)