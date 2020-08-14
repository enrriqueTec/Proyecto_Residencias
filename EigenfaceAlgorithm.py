# OpenCV module
import cv2
# Modulo para leer directorios y rutas de archivos
import os
# OpenCV trabaja con arreglos de numpy
import numpy
from time import time

from listaPermitidos import flabianos

flabs = flabianos()

# iniciamos el tiempo a medir
starting_point = time()
# Parte 1: Creando el entrenamiento del modelo
# Directorio donde se encuentran las carpetas con las caras de entrenamiento
dir_faces = 'att_faces/orl_faces'
# Tamaño para reducir a miniaturas las fotografias
size = 4

# Crear una lista de imagenes y una lista de nombres correspondientes
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(dir_faces):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dir_faces, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(im_width, im_height) = (112, 92)

# Crear una matriz Numpy de las dos listas anteriores
(images, lables) = [numpy.array(lis) for lis in [images, lables]]
# OpenCV entrena un modelo a partir de las imagenes
model = cv2.face.EigenFaceRecognizer_create(num_components=200)
model.train(images, lables)

# Parte 2: Utilizar el modelo entrenado en funcionamiento con la camara
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

frame=cv2.imread("att_faces/Prueba/20.png")
frame=cv2.resize(frame,(268,308))

0# convertimos la imagen a blanco y negro
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# redimensionar la imagen
mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

faces = face_cascade.detectMultiScale(mini)

retval=	cv2.face_FaceRecognizer.predict_label

while True:
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]

        # Obteniendo rostro
        face_img = frame[y:y + h, h:h + w].copy()


        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))



        # Intentado reconocer la cara
        prediction = model.predict(face_resize)

        #theshold=cv2.THRESH_BINARY
        #ret, img_binary=cv2.threshold(face_resize,150,255,theshold)
        #retval = model.predict_label(face_resize)
        #print()

        # Dibujamos un rectangulo en las coordenadas del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Escribiendo el nombre de la cara reconocida
        # La variable cara tendra el nombre de la persona reconocida
        cara = '%s' % (names[prediction[0]])

        # Si la prediccion tiene una exactitud menor a 100 se toma como prediccion valida
        print(prediction[1])

        if prediction[1] >2700 :
            # En caso de que la cara sea de algun conocido se realizara determinadas accione
            # Busca si los nombres de las personas reconocidas estan dentro de los que tienen acceso
            flabs.valida_invitado(cara)
            # Ponemos el nombre de la persona que se reconoció
            cv2.putText(frame, '%s : %.0f ' % (cara, prediction[1]),
                        (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))


        elif prediction[1] < 2700 :

            # Si la prediccion es mayor a 3000 no es un reconomiento con la exactitud suficiente
            # Si la cara es desconocida, poner desconocido
            cv2.putText(frame, 'Desconocido', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        # Mostramos la imagen
    cv2.imshow('Eigenface', frame)
    #cv2.imshow("imagen despues del th",img_binary)
    # Medir tiempo de ejecucion
    elapsed_time = time() - starting_point
    elapsed_time_int = int(elapsed_time)
    print(elapsed_time_int)
    # Si se presiona la tecla ESC se cierra el programa
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyAllWindows()
        break