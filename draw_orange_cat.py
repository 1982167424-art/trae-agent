from PIL import Image, ImageDraw

# Create a new image with white background
width, height = 400, 400
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Define colors
orange = '#FFA500'  # Orange color for the cat's body
light_orange = '#FFB84D'  # Lighter orange for ears
pink = '#FFC0CB'  # Pink for nose
black = '#000000'  # Black for eyes, whiskers, mouth
white = '#FFFFFF'  # White for eyes

# Draw cat body (ellipse)
body_x1, body_y1 = 100, 150
body_x2, body_y2 = 300, 350
draw.ellipse([body_x1, body_y1, body_x2, body_y2], fill=orange, outline=orange)

# Draw left ear (triangle)
left_ear = [(130, 100), (100, 150), (160, 150)]
draw.polygon(left_ear, fill=light_orange, outline=orange)

# Draw right ear (triangle)
right_ear = [(270, 100), (240, 150), (300, 150)]
draw.polygon(right_ear, fill=light_orange, outline=orange)

# Draw left eye
left_eye_center = (160, 200)
draw.ellipse([left_eye_center[0]-20, left_eye_center[1]-20, left_eye_center[0]+20, left_eye_center[1]+20], fill=white, outline=black)
# Left eye pupil
draw.ellipse([left_eye_center[0]-8, left_eye_center[1]-8, left_eye_center[0]+8, left_eye_center[1]+8], fill=black)

# Draw right eye
right_eye_center = (240, 200)
draw.ellipse([right_eye_center[0]-20, right_eye_center[1]-20, right_eye_center[0]+20, right_eye_center[1]+20], fill=white, outline=black)
# Right eye pupil
draw.ellipse([right_eye_center[0]-8, right_eye_center[1]-8, right_eye_center[0]+8, right_eye_center[1]+8], fill=black)

# Draw nose
nose_center = (200, 250)
nose_size = 10
nose = [(nose_center[0], nose_center[1]-nose_size), 
        (nose_center[0]-nose_size, nose_center[1]+nose_size), 
        (nose_center[0]+nose_size, nose_center[1]+nose_size)]
draw.polygon(nose, fill=pink, outline=black)

# Draw mouth
draw.line([(200, 260), (200, 275)], fill=black, width=2)
draw.arc([(190, 270), (210, 280)], 180, 0, fill=black, width=2)

# Draw whiskers
# Left whiskers
draw.line([(150, 250), (80, 240)], fill=black, width=1)
draw.line([(150, 260), (80, 260)], fill=black, width=1)
draw.line([(150, 270), (80, 280)], fill=black, width=1)

# Right whiskers
draw.line([(250, 250), (320, 240)], fill=black, width=1)
draw.line([(250, 260), (320, 260)], fill=black, width=1)
draw.line([(250, 270), (320, 280)], fill=black, width=1)

# Save the image
save_path = '/Users/zhang/Desktop/tim/橘猫-4.0.png'
image.save(save_path)
print(f"橘猫图片已成功保存到: {save_path}")
