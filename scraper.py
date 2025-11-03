import tkinter as tk
from tkinter import messagebox, simpledialog
import instaloader
import pandas as pd
import re
import os

def extract_shortcode(link: str):
    match = re.search(r"instagram\.com/(?:p|reel|tv)/([A-Za-z0-9_-]+)/?", link)
    return match.group(1) if match else None

def scrape_post(link: str, output_csv="instagram_data.csv"):
    shortcode = extract_shortcode(link)
    if not shortcode:
        messagebox.showerror("Error", "Invalid Instagram link.")
        return

    L = instaloader.Instaloader()

    # Ask username
    username = simpledialog.askstring("Login", "Enter your Instagram username:")
    if not username:
        messagebox.showwarning("Cancelled", "No username provided.")
        return

    session_file = f"{username}.session"
    try:
        if os.path.exists(session_file):
            L.load_session_from_file(username, session_file)
            messagebox.showinfo("Session", "✅ Loaded saved session.")
        else:
            password = simpledialog.askstring("Login", "Enter your Instagram password:", show="*")
            if not password:
                messagebox.showwarning("Cancelled", "Password not entered.")
                return
            L.login(username, password)
            L.save_session_to_file(session_file)
            messagebox.showinfo("Session", "🔐 Logged in and session saved.")
    except Exception as e:
        messagebox.showerror("Login Error", f"Login failed: {e}")
        return

    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except Exception as e:
        messagebox.showerror("Error", f"Could not fetch post: {e}")
        return

    caption = post.caption or ""
    hashtags = post.caption_hashtags
    likes = post.likes
    date = post.date_utc
    post_type = "Video" if post.is_video else "Image/Carousel"
    url = f"https://www.instagram.com/p/{shortcode}/"

    comments = []
    try:
        for i, c in enumerate(post.get_comments()):
            if i >= 10:
                break
            comments.append(f"{c.owner.username}: {c.text}")
    except Exception as e:
        comments.append(f"⚠️ Could not access comments: {e}")

    data = {
        "Post Type": [post_type],
        "Caption": [caption],
        "Hashtags": [", ".join(hashtags)],
        "Likes": [likes],
        "Date": [date],
        "URL": [url],
        "Top 10 Comments": [" | ".join(comments) if comments else "None"]
    }

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    messagebox.showinfo(
        "Scraping Complete",
        f"✅ Data saved to '{output_csv}'\n\n"
        f"Likes: {likes}\nHashtags: {', '.join(hashtags) or 'None'}\n\n"
        f"Caption:\n{caption[:200]}..."
    )

def on_scrape():
    link = entry_link.get().strip()
    if not link:
        messagebox.showwarning("Missing Input", "Please paste an Instagram post/reel link first.")
        return
    scrape_post(link)

# # GUI setup
# root = tk.Tk()
# root.title("Instagram Post Scraper")
# root.geometry("400x220")
# root.resizable(False, False)

# tk.Label(root, text="📸 Instagram Scraper", font=("Arial", 16, "bold")).pack(pady=10)
# tk.Label(root, text="Paste your Instagram post or reel link below:", font=("Arial", 10)).pack()

# entry_link = tk.Entry(root, width=50, font=("Arial", 10))
# entry_link.pack(pady=5)

# btn_scrape = tk.Button(root, text="Scrape Post", command=on_scrape, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
# btn_scrape.pack(pady=10)

# tk.Label(root, text="Result will be saved to instagram_data.csv", font=("Arial", 9, "italic")).pack(pady=5)

# root.mainloop()
