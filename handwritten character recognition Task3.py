from tkinter import *
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
from tkinter import filedialog

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([

    tf.keras.layers.Reshape(
        (28, 28, 1),
        input_shape=(28, 28)
    ),

    tf.keras.layers.Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=3
)


root = Tk()
root.title("Handwritten Character Recognition")
root.geometry("950x700")
root.config(bg="#0f172a")

def predict_digit():

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if file_path:

        image = Image.open(file_path)

        image = image.convert("L")

        image = ImageOps.invert(image)

        image = image.resize((28,28))

        image_array = np.array(image) / 255.0

        image_array = image_array.reshape(
            1,
            28,
            28
        )

        prediction = model.predict(
            image_array,
            verbose=0
        )

        digit = np.argmax(prediction)

        result_label.config(
            text=f"Predicted Character: {digit}",
            fg="#22c55e"
        )


top_frame = Frame(
    root,
    bg="#111827",
    height=100
)

top_frame.pack(fill="x")

Label(
    top_frame,
    text="HANDWRITTEN CHARACTER RECOGNITION",
    font=("Arial", 28, "bold"),
    bg="#111827",
    fg="#38bdf8"
).pack(pady=25)


main_frame = Frame(
    root,
    bg="#1e293b"
)

main_frame.pack(
    pady=60,
    ipadx=50,
    ipady=40
)


Label(
    main_frame,
    text="Upload Handwritten Digit Image",
    font=("Arial", 22, "bold"),
    bg="#1e293b",
    fg="white"
).pack(pady=20)


Button(
    main_frame,
    text="UPLOAD IMAGE",
    command=predict_digit,
    font=("Arial", 16, "bold"),
    bg="#06b6d4",
    fg="black",
    padx=25,
    pady=12,
    bd=0,
    cursor="hand2"
).pack(pady=25)

result_label = Label(
    root,
    text="",
    font=("Arial", 32, "bold"),
    bg="#0f172a"
)

result_label.pack(pady=40)

Label(
    root,
    text="Developed Using CNN & TensorFlow",
    font=("Arial", 12),
    bg="#0f172a",
    fg="#94a3b8"
).pack(side=BOTTOM, pady=15)

root.mainloop()
