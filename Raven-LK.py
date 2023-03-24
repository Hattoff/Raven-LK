import re
import os
import json
import openai
import tkinter as tk
from time import time, sleep
from threading import Thread
from tkinter import ttk, scrolledtext
from spellchecker import SpellChecker
import screeninfo
from ConversationManagement import ConversationManager
conversation_manager = ConversationManager()
_full_height = False
_window_width = 600
_window_height = 600

####### TKINTER functions by GPT4 prompted by David Shapiro(daveshap); modified by Matt Hatton(hattoff)
####### https://github.com/daveshap/Chapter_Summarizer_GPT4/blob/main/chat_tkinter2.py

def send_message(event=None):
    user_input = user_entry.get("1.0", "end-1c")
    if not user_input.strip():
        return
    user_entry.delete("1.0", "end")
    display_user_response(user_input)
    chat_text.config(state='disabled')
    ai_status.set("Input is processing...")
    conversation_manager.log_message('USER', user_input)
    
    ai_status.set("Raven is thinking...")
    Thread(target=get_ai_response).start()

## Display the message from the user, if the timestamp has data then it will display 
## the "summary" version which is italic and append the timestamp to the speaker tag
def display_user_response(user_input, timestamp = ''):
    if timestamp != '':
        tag_name = 'user-summary'
        user_response = f"\nUSER [{timestamp}]:\n{user_input}\n\n"
    else:
        tag_name = 'user'
        user_response = f"\nUSER:\n{user_input}\n\n"
    chat_text.config(state='normal')
    chat_text.insert(tk.END, user_response, tag_name)
    chat_text.see(tk.END)

def get_ai_response():
    ## GPT Response
    raven_response = conversation_manager.generate_response()
    conversation_manager.log_message('RAVEN', raven_response)
    display_ai_response(raven_response)
    ai_status.set("")

## Display the response from the AI, if the timestamp has data then it will display 
## the "summary" version which is italic and append the timestamp to the speaker tag
def display_ai_response(response, timestamp = ''):
    if timestamp != '':
        tag_name = 'raven-summary'
        ai_response = f"\nRAVEN [{timestamp}]:\n{response}\n\n"
    else:
        tag_name = 'raven'
        ai_response = f"\nRAVEN:\n{response}\n\n"
    chat_text.config(state='normal')
    chat_text.insert(tk.END, ai_response, tag_name)
    chat_text.see(tk.END)
    chat_text.config(state='disabled')

## Drop a new line when shift+enter is pressed, otherwise send the current message
def on_return_key(event):
    if event.state & 0x1:  # Shift key is pressed
        user_entry.insert(tk.END, '\n')
    else:
        send_message()

## Initialize the conversation and display the most recent messages for context
def load_conversation(message_history_count = 2):
    recent_messages = conversation_manager.load_state(message_history_count)
    message_count = len(recent_messages)
    if message_count == 0:
        chat_text.insert(tk.END, "\nWelcome!\n\n", 'system')
        return
    else:
        chat_text.insert(tk.END, "\nWelcome Back!\n\n", 'system')
    for m in range(message_count-1):
        if recent_messages[m]['speaker'] == 'USER':
            display_user_response(recent_messages[m]['content'],recent_messages[m]['timestring'])
        else:
            display_ai_response(recent_messages[m]['content'],recent_messages[m]['timestring'])
    if recent_messages[message_count-1]['speaker'] == 'USER':
        user_entry.insert("1.0", recent_messages[message_count-1]['content'])
    else:
        display_ai_response(recent_messages[message_count-1]['content'],recent_messages[message_count-1]['timestring'])

## Set the center of the window to the user's cursor
def snap_window_to_cursor(window_width = 400, window_height = 400):
    # Get the position of the user's cursor
    cursor_x = root.winfo_pointerx()
    cursor_y = root.winfo_pointery()

    # Calculate the new position of the window
    window_x = cursor_x - (window_width // 2)
    window_y = cursor_y - (window_height // 2)

    global _window_width
    global _window_height

    _window_width = window_width
    _window_height = window_height

    root.geometry("{0}x{1}+{2}+{3}".format(window_width, window_height, window_x, window_y))

# Toggle the overrideredirect method of the root window when F11 is pressed
def toggle_fullscreen(event):
    global _full_height
    global _window_width
    global _window_height

    if _full_height:
        root.overrideredirect(False)
        _full_height = False

    if root.attributes("-fullscreen"):
        root.attributes("-fullscreen", False)
        root.overrideredirect(False)
        snap_window_to_cursor(_window_width, _window_height)
    else:
        window_width = root.winfo_width()
        _window_width = window_width

        window_height = root.winfo_height()
        _window_height = window_height

        root.attributes("-fullscreen", True)
        root.overrideredirect(True)

# Toggle the full height of the root window when F10 is pressed
def toggle_fullheight(event):
    global _full_height
    global _window_width
    global _window_height

    if _full_height:
        root.overrideredirect(False)
        snap_window_to_cursor(_window_width, _window_height)
        _full_height = False
    else:
        screen_info = get_screen_info_at_cursor()
        
        window_width = root.winfo_width()
        _window_width = window_width

        window_height = root.winfo_height()
        _window_height = window_height

        cursor_x = root.winfo_pointerx()

        # Set the x coordinate of the window geometry to the distance to the closest edge
        if (cursor_x - screen_info['left']) < screen_info['width'] // 2:
            x_coord = screen_info['left']
        else:
            x_coord = (screen_info['width']+screen_info['left']) - window_width
        root.geometry("{0}x{1}+{2}+0".format(window_width, screen_info['height'], x_coord))
        root.overrideredirect(True)
        _full_height = True

def get_screen_info_at_cursor():
    # Get the x and y coordinates of the cursor
    x = root.winfo_pointerx()
    y = root.winfo_pointery()

    # Find the screen that contains the cursor
    screen = None
    for s in screeninfo.get_monitors():
        if s.x <= x < s.x + s.width and s.y <= y < s.y + s.height:
            screen = s
            break

    # If no screen was found, return None
    if screen is None:
        return None

    # Return the screen information
    return {
        'width': screen.width,
        'height': screen.height,
        'left': screen.x,
        'top': screen.y,
    }

# Define a function to check the spelling of the text in the text box
def check_spelling(event):
    ## Don't check spelling until the end of a word or sentence to avoid false positives
    punctuations = [",", ".", ";", ":", "?", "!", " ", "\n"]
    if event.char not in punctuations:
        return

    user_entry.tag_remove('spell_error', '1.0', 'end')
    # user_entry.tag_configure('spell_error', background='red')
    user_entry.tag_configure('spell_error', underline=True, font=("Calibri", 12, 'italic'))
    # Get the contents of the text box and split into words
    text = user_entry.get('1.0', 'end-1c')
    words = text.split()

    # Check the spelling of each word and display suggestions for misspelled words
    for word in words:
        # Remove the punctuation characters from the string to avoid false positives
        clean_word = ''.join(c for c in word if c not in punctuations)
        if not clean_word:
            continue
        if not spell.unknown([clean_word]):
            continue

        # Get suggestions for misspelled words
        start_position = user_entry.search(word, '1.0', stopindex=tk.END)
        line, column = start_position.split(".")
        end_position = f'{line}.{str(int(column)+len(clean_word))}'

        # Display suggestions for misspelled words
        user_entry.tag_add("spell_error", start_position, end_position)


if __name__ == "__main__":
    # Tkinter GUI
    root = tk.Tk()
    root.title("Raven-LK")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=1)

    chat_text = tk.Text(main_frame, wrap=tk.WORD, width=60, height=20, bg='#333333')
    chat_text.grid(column=0, row=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
    chat_text.tag_configure('user', background='#444654', wrap='word', justify='right',foreground="white", font=("Calibri", 12))
    chat_text.tag_configure('user-summary', background='#444654', wrap='word', justify='right', font=("Calibri", 12, "italic"),foreground="white")
    chat_text.tag_configure('raven', background='#343541', wrap='word', justify='left',foreground="white", font=("Calibri", 12))
    chat_text.tag_configure('raven-summary', background='#343541', wrap='word', justify='left', font=("Calibri", 12, "italic"),foreground="white")
    chat_text.tag_configure('system', justify='center', foreground="white", font=("Calibri", 14, "bold"))

    send_button = ttk.Button(main_frame, text="Send", command=send_message)
    send_button.grid(column=1, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))

    ai_status = tk.StringVar()
    ai_status_label = ttk.Label(main_frame, textvariable=ai_status)
    ai_status_label.grid(column=2, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Initialize the text box
    user_entry = tk.Text(main_frame, width=50, height=5, wrap="word", font=("Calibri", 12))
    user_entry.grid(column=0, row=1, sticky=(tk.W, tk.E))
    # Initialize the spell checker
    spell = SpellChecker()

    user_entry.focus()
    root.bind("<Return>", on_return_key)

    # Bind the F11 key to the toggle_fullscreen function
    root.bind("<F11>", toggle_fullscreen)

    # Bind the F10 key to the toggle_fullheight function
    root.bind("<F10>", toggle_fullheight)
    
    # Bind the <KeyRelease> event to the check_spelling function
    root.bind('<KeyRelease>', check_spelling)
    
    load_conversation(10)
    chat_text.config(state='disabled')

    snap_window_to_cursor(600, 600)
    root.mainloop()