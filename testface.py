# import face_recognition
# import cv2
# import pyttsx3
# import time

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()

# # Load known images and encode their faces
# known_people = {
#     "ayaan": face_recognition.face_encodings(
#         face_recognition.load_image_file(
#             "C:\\Users\\asus\\Desktop\\OSKI\\images\\pho.jpg"
#         )
#     )[0]
# }

# # Set the desired frame rate
# desired_frame_rate = 30

# # Capture video from your webcam (you may need to adjust the camera index)
# video_capture = cv2.VideoCapture(0)
# video_capture.set(cv2.CAP_PROP_FPS, desired_frame_rate)

# # Define light blue color for border and black color for inner box
# border_color = (255, 192, 203)  # Light blue color (RGB)
# inner_box_color = (0, 0, 0)  # Black color (RGB)

# # Flag to track whether face has been recognized
# face_recognized = False

# # Timer for animation display
# animation_timer = 0

# # Flag to control window closing after animation
# close_window = False

# while True:
#     # Capture each frame
#     _, frame = video_capture.read()

#     # Find all face locations and face encodings in the current frame
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)

#     # If face has already been recognized or animation is ongoing, skip face recognition
#     if not face_recognized and animation_timer == 0:
#         # Loop through each face found in the frame
#         for (top, right, bottom, left), face_encoding in zip(
#             face_locations, face_encodings
#         ):
#             # Check if the face matches any known faces
#             matches = face_recognition.compare_faces(
#                 list(known_people.values()), face_encoding
#             )

#             for idx, match in enumerate(matches):
#                 if match:
#                     name = list(known_people.keys())[idx]
#                     text = "Face Recognized: " + name

#                     # Draw a rectangle around the face
#                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

#                     # Draw the name label below the face
#                     cv2.putText(
#                         frame,
#                         text,
#                         (left, bottom + 20),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (0, 255, 0),
#                         1,
#                         cv2.LINE_AA,
#                     )

#                     # Set the flag to True
#                     face_recognized = True

#                     # Say "Face Recognized"
#                     engine.say("Face Recognized")
#                     engine.runAndWait()

#                     # Start animation timer
#                     animation_timer = time.time()

#                     break  # Break the loop if a match is found

#     # If animation is ongoing, display the animation for 2 seconds
#     if animation_timer != 0:
#         elapsed_time = time.time() - animation_timer
#         if elapsed_time < 2:
#             # Display unlocking animation (e.g., drawing a rectangle from left to right)
#             animation_progress = elapsed_time / 2  # Progress from 0 to 1
#             unlock_width = int(animation_progress * frame.shape[1])
#             cv2.rectangle(
#                 frame, (0, 0), (unlock_width, frame.shape[0]), (255, 192, 203), -1
#             )
#         else:
#             # Animation finished, set flag to close window
#             close_window = True

#     # Add a light blue border around the frame
#     frame_with_border = cv2.copyMakeBorder(
#         frame,
#         top=10,
#         bottom=10,
#         left=10,
#         right=10,
#         borderType=cv2.BORDER_CONSTANT,
#         value=border_color,
#     )

#     # Draw a black box to place "Face Recognition" text
#     (text_width, text_height), _ = cv2.getTextSize(
#         "Face Recognition", cv2.FONT_HERSHEY_SIMPLEX, 1, 2
#     )
#     cv2.rectangle(
#         frame_with_border,
#         (10, 10),
#         (10 + text_width + 20, 10 + text_height + 20),
#         inner_box_color,
#         -1,  # Fill the rectangle
#     )

#     # Add text "Face Recognition" inside the inner box
#     cv2.putText(
#         frame_with_border,
#         "Face Recognition",
#         (20, 40),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (255, 255, 255),  # White color for text
#         2,
#         cv2.LINE_AA,
#     )

#     # Display the resulting frame with border
#     cv2.imshow("Video", frame_with_border)

#     # Check if window should be closed
#     if close_window:
#         break

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(5) & 0xFF == ord("q"):
#         break

# # Release the video capture object and close the window
# video_capture.release()
# cv2.destroyAllWindows()
import face_recognition
import cv2
import pyttsx3
import time


def face_recognition_with_animation(image_path):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Load known images and encode their faces
    known_people = {
        "ayaan": face_recognition.face_encodings(
            face_recognition.load_image_file(image_path)
        )[0]
    }

    # Set the desired frame rate
    desired_frame_rate = 30

    # Capture video from your webcam (you may need to adjust the camera index)
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FPS, desired_frame_rate)

    # Define light blue color for border and black color for inner box
    border_color = (255, 192, 203)  # Light blue color (RGB)
    inner_box_color = (0, 0, 0)  # Black color (RGB)

    # Flag to track whether face has been recognized
    face_recognized = False

    # Timer for animation display
    animation_timer = 0

    # Flag to control window closing after animation
    close_window = False

    while True:
        # Capture each frame
        _, frame = video_capture.read()

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # If face has already been recognized or animation is ongoing, skip face recognition
        if not face_recognized and animation_timer == 0:
            # Loop through each face found in the frame
            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                # Check if the face matches any known faces
                matches = face_recognition.compare_faces(
                    list(known_people.values()), face_encoding
                )

                for idx, match in enumerate(matches):
                    if match:
                        name = list(known_people.keys())[idx]
                        text = "Face Recognized: " + name

                        # Draw a rectangle around the face
                        cv2.rectangle(
                            frame, (left, top), (right, bottom), (0, 255, 0), 2
                        )

                        # Draw the name label below the face
                        cv2.putText(
                            frame,
                            text,
                            (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                        # Set the flag to True
                        face_recognized = True

                        # Say "Face Recognized"
                        engine.say("Face Recognized")
                        engine.runAndWait()

                        # Start animation timer
                        animation_timer = time.time()

                        break  # Break the loop if a match is found

        # If animation is ongoing, display the animation for 2 seconds
        if animation_timer != 0:
            elapsed_time = time.time() - animation_timer
            if elapsed_time < 2:
                # Display unlocking animation (e.g., drawing a rectangle from left to right)
                animation_progress = elapsed_time / 2  # Progress from 0 to 1
                unlock_width = int(animation_progress * frame.shape[1])
                cv2.rectangle(
                    frame, (0, 0), (unlock_width, frame.shape[0]), (255, 192, 203), -1
                )
            else:
                # Animation finished, set flag to close window
                close_window = True

        # Add a light blue border around the frame
        frame_with_border = cv2.copyMakeBorder(
            frame,
            top=10,
            bottom=10,
            left=10,
            right=10,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )

        # Draw a black box to place "Face Recognition" text
        (text_width, text_height), _ = cv2.getTextSize(
            "Face Recognition", cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )
        cv2.rectangle(
            frame_with_border,
            (10, 10),
            (10 + text_width + 20, 10 + text_height + 20),
            inner_box_color,
            -1,  # Fill the rectangle
        )

        # Add text "Face Recognition" inside the inner box
        cv2.putText(
            frame_with_border,
            "Face Recognition",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),  # White color for text
            2,
            cv2.LINE_AA,
        )

        # Display the resulting frame with border
        cv2.imshow("Video", frame_with_border)

        # Check if window should be closed
        if close_window:
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the window
    video_capture.release()
    cv2.destroyAllWindows()


# Example usage:
# face_recognition_with_animation("C:\\Users\\asus\\Desktop\\OSKI\\images\\pho.jpg")
