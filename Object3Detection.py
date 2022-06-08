import streamlit as st
import cv2
import mediapipe as mp
import time



def main():


    st.title("3D Object Detection Web App")
    st.sidebar.title("Functions")
    st.markdown("A simple web app to detect simple objects based on Objectron")
    st.markdown("Current supported models: Chair, Cup, Shoe")
    st.sidebar.markdown("Object Detection Parameters")

    run = st.checkbox('Open Webcam')
    cap = cv2.VideoCapture(0)


    
    my_placeholder = st.empty()

    st.write("")
    st.markdown("For documentation of MediaPipe: https://google.github.io/mediapipe/solutions/objectron.html")

    mp_objectron = mp.solutions.objectron
    mp_drawing = mp.solutions.drawing_utils

    st.sidebar.subheader("Choose Object types")
    objects = st.sidebar.selectbox("Objects", ("Cup", "Chair", "Shoe"))

    if objects == 'Cup':
        model_name = 'Cup'
        st.sidebar.subheader("Adjusting parameters")
        max_num_objects = st.sidebar.slider("Maximum number of objects", 1, 5, key='max1')
        min_detection_confidence = st.sidebar.slider("Minimum detection confidence", 0.4, float(1),  key='mind1')
        min_tracking_confidence = st.sidebar.slider("Minimum tracking confidence", 0.7, float(1), key='mint1')

    if objects == 'Chair':
        model_name = 'Chair'
        st.sidebar.subheader("Adjusting parameters")
        max_num_objects = st.sidebar.slider("Maximum number of objects", 1, 5, key='max1')
        min_detection_confidence = st.sidebar.slider("Minimum detection confidence", 0.4, float(1),  key='mind1')
        min_tracking_confidence = st.sidebar.slider("Minimum tracking confidence", 0.7, float(1), key='mint1')

    if objects == 'Shoe':
        model_name = 'Shoe'
        st.sidebar.subheader("Adjusting parameters")
        max_num_objects = st.sidebar.slider("Maximum number of objects", 1, 5, key='max1')
        min_detection_confidence = st.sidebar.slider("Minimum detection confidence", 0.4, float(1),  key='mind1')
        min_tracking_confidence = st.sidebar.slider("Minimum tracking confidence", 0.7, float(1), key='mint1')




    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=max_num_objects,
                                min_detection_confidence=min_detection_confidence,
                                min_tracking_confidence=min_tracking_confidence,
                                model_name=model_name) as objectron:
        while run:

            while cap.isOpened():

                success, image = cap.read()
                start = time.time()

                # convert BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # for performance can turn to False
                image.flags.writeable = False
                results = objectron.process(image)

                # image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.detected_objects:
                    for detected_objects in results.detected_objects:
                        mp_drawing.draw_landmarks(image, detected_objects.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                        mp_drawing.draw_axis(image, detected_objects.rotation, detected_objects.translation)

                end = time.time()
                totaltime = end - start

                fps = 1 / totaltime

                cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                my_placeholder.image(image, use_column_width=True)
                # cv2.imshow('MediaPipe Objectron', image)

                # if cv2.waitKey(5) & 0xFF == 27:
                #     break

    cap.release()


if __name__ == '__main__':
    main()