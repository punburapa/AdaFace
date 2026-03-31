# check_path.py
import os

LFW_ROOT = "data/lfw"

# เช็คว่า folder มีอยู่ไหม
print(f"Exists: {os.path.exists(LFW_ROOT)}")

# ดูว่ามีโฟลเดอร์อะไรบ้าง
if os.path.exists(LFW_ROOT):
    folders = os.listdir(LFW_ROOT)
    print(f"Total folders: {len(folders)}")
    print(f"Sample folders: {folders[:5]}")
    
    # ดูไฟล์ใน folder แรก
    first = folders
    files = os.listdir(os.path.join(LFW_ROOT, first))
    print(f"\nSample files in {first}:")
    print(files)
else:
    print("❌ ไม่เจอ folder!")
    print("📂 มีอะไรใน data/ บ้าง:")
    if os.path.exists("data"):
        print(os.listdir("data"))
    else:
        print("ไม่มีโฟลเดอร์ data/ เลย!")