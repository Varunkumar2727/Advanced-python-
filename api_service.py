import cv2
import webbrowser
import urllib.parse

# Get source and destination from the user
source = input("Enter source location: ")
destination = input("Enter destination location: ")

# Convert locations into a URL-safe format
source_encoded = urllib.parse.quote(source)
destination_encoded = urllib.parse.quote(destination)

# Create Google Maps directions URL
google_maps_url = (
    f"https://www.google.com/maps/dir/?api=1"
    f"&origin={source_encoded}"
    f"&destination={destination_encoded}"
)

# Open Google Maps in the default browser
webbrowser.open(google_maps_url)

# OpenCV window
image = 255 * __import__("numpy").ones((300, 700, 3), dtype="uint8")

cv2.putText(
    image,
    "Google Maps Opened!",
    (170, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 0, 0),
    2
)

cv2.putText(
    image,
    "Press any key to close",
    (190, 180),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 0),
    2
)

cv2.imshow("Google Maps", image)

cv2.waitKey(0)
cv2.destroyAllWindows()