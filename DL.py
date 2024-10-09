import cv2
import os
import time

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_images(person_name, save_dir='dataset', num_images=100, delay=1):
    # Create directory for the person
    person_dir = os.path.join(save_dir, person_name)
    create_directory(person_dir)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()  # Capture frame from the webcam
        if not ret:
            print(f"Failed to capture image for {person_name}.")
            break

        # Display the frame
        cv2.imshow(f'Capturing Images for {person_name} - Press "q" to quit', frame)

        # Save the image to the person's folder
        img_name = os.path.join(person_dir, f'{person_name}_{count}.jpg')
        cv2.imwrite(img_name, frame)
        print(f"Image {count+1}/{num_images} captured and saved as {img_name}")

        count += 1

        # Introduce a delay (in seconds) between captures
        time.sleep(delay)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

def capture_for_multiple_people(people, save_dir='dataset', num_images=100, delay=1):
    for person in people:
        print(f"\nStarting capture for {person}")
        capture_images(person_name=person, save_dir=save_dir, num_images=num_images, delay=delay)
        print(f"Finished capturing images for {person}\n")

# Example usage:
people_names = input("Enter names of people (comma-separated): ").split(",")  # Multiple names
num_images = 5  # Capture 5 images per person
delay = 2  # 2-second delay between captures

# Trim whitespace from each name
people_names = [name.strip() for name in people_names]

# Capture images for each person
capture_for_multiple_people(people_names, num_images=num_images, delay=delay)