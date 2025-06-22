import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = 'models/vgg16_transfer_learning_model.h5'
IMAGE_SIZE = (150, 150)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("モデルのロードに成功")
except Exception as e:
    messagebox.showerror("エラー", f"モデルのロードに失敗: {e}")
    model = None



###
class DogCatClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("犬猫識別アプリ")
        self.root.geometry("600x500")
        self.image_path = None

        self.title_label = tk.Label(root, text="犬か猫の画像をアップロード", font=("Helvetica", 16))
        self.title_label.pack(pady=20)

        self.image_frame = tk.Frame(root, width=400, height=300, bd=2, relief="groove")
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="結果：", font=("Helvetica", 14, "bold"))
        self.result_label.pack(pady=10)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=20)
        self.select_button = tk.Button(self.button_frame, text="画像を選択", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=10)
        self.classify_button = tk.Button(self.button_frame, text="識別", command=self.classify_image, state=tk.DISABLED)
        self.classify_button.pack(side=tk.LEFT, padx=10)



###
    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.classify_button.config(state=tk.NORMAL)
            self.result_label.config(text="結果：")

    def display_image(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((self.image_frame.winfo_width() - 20, self.image_frame.winfo_height() - 20), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img
        except Exception as e:
            messagebox.showerror("エラー", f"画像の表示に失敗（通常はあり得ないはず）: {e}")

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array



###
    def classify_image(self):
        if not model:
            messagebox.showerror("エラー", "モデルがロードされていません（一応）")
            return
        if self.image_path:
            try:
                preprocessed_img = self.preprocess_image(self.image_path)
                prediction = model.predict(preprocessed_img)[0][0]
                if prediction > 0.5:
                    result_text = f"結果：犬 ({prediction*100:.2f}%)"
                    self.result_label.config(fg="blue")
                else:
                    result_text = f"結果：猫 ({(1-prediction)*100:.2f}%)"
                    self.result_label.config(fg="green")
                self.result_label.config(text=result_text)
            except Exception as e:
                messagebox.showerror("エラー")
        else:
            messagebox.showwarning("画像をアップロードしてください")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        root = tk.Tk()
        app = DogCatClassifierApp(root)
        root.mainloop()
    else:
        messagebox.showerror("エラー", f"モデルファイル '{MODEL_PATH}' が見つかりません")
