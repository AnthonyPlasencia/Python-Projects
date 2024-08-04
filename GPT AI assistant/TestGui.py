from tkinter import *

root = Tk()
root.title("Test GUI")
root.geometry("400x400")
root.configure(bg="black", padx=10, pady=10)
UserInput = Entry(root, width=50, bg="black", fg="white", borderwidth=5)

def click():
    myLabel = Label(root, text=UserInput.get())
    myLabel.pack()

Button1 = Button(root, text="Click me!", padx=50, pady=50,command=click)

Button1.pack()
UserInput.pack()

root.mainloop()
