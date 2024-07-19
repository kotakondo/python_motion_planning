import cv2
import os

def make_video_from_images(image_folder, output_video, fps=3):
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # sort the images but make sure figure10.png comes after figure9.png
    images.sort(key=lambda x: int(x.split("figure")[1].split(".png")[0]))

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the VideoWriter
    video.release()
    print(f"Video saved as {output_video}")

def main():
    image_folder = '/home/kkondo/Downloads/tmp'
    output_video = image_folder + '/output_video.mp4'
    make_video_from_images(image_folder, output_video)

if __name__ == "__main__":
    main()
