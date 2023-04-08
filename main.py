import face_checker as fc 
import face_extract as fe 
import gradio as gr
from PIL import Image
def fun(id_img_path,selfie_img_path):
   img = Image.fromarray(id_img_path,'RGB')
   img.save("id.jpg")
   img2 = Image.fromarray(selfie_img_path,'RGB')
   img2.save("selfie.jpg")
   extracted_facename = fe.extract("id.jpg")
   res = fc.check(extracted_facename,'selfie.jpg')
   return str(res)

interface = gr.Interface(fun,inputs=[gr.Image(),gr.Image()],outputs='text')
   # Will return True or false


interface.launch()

# id_img_path ="test.jpg" #give image path
# selfie_img_path = "142142_faces.jpg" #give path of curret face