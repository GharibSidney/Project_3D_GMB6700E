from heic2png import HEIC2PNG
import os
if __name__ == '__main__':
    for image in os.listdir('image_Luigi/heic/'):
        if image.endswith('.heic') or image.endswith('.HEIC'):
            heic_img = HEIC2PNG("image_Luigi/heic/"+image, quality=100)  # Specify the quality of the converted image
            png_image_name = os.path.splitext(image)[0] + ".png"
            heic_img.save("image_Luigi/png/"+ png_image_name, extension=".png")  # Save the converted image with .png extension
            print(f"Converted {image} to {png_image_name}")
    print("Done converting!")
    # heic_img = HEIC2PNG('test.heic', quality=100)  # Specify the quality of the converted image
    # heic_img.save()  # The converted image will be saved as `test.png`