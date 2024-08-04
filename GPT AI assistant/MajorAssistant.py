
from tkinter import *
import os
from openai import OpenAI
client = OpenAI()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# Read the content of the "majors" text file
def read_majors_file(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return None
    except Exception as e:
        print("Error:", e)
        return None
    
#get current directory path that we are in
current_directory = os.path.dirname(os.path.realpath(__file__))
majors_file = os.path.join(current_directory, "majors.txt")

newInstructions = ("You are an AI chat bot that will help a student decide on a major based on their hobbies and interests."
                   + "First ask the user for any hobbies they have, then ask for any interests they have."
                   + "After, ask if they have any strong dislikes to narrow down the major options to choose from. "
                   + "Finally ask for any other notes they may have. Then use all this information to recommend a major. "
                   + "Finally when giving the major recommended, give them at least 3. These majors are recommended in the 'majors.txt' file and should only give options from those." 
                   + "The one that's recommended the most should give an explanation on why it might fit the user the best, while the other two should just be bullet points."
                   + "The name of the school you operate for as well as all of its majors provided are: " 
                   + read_majors_file(majors_file) 
                   + "You may only provide recommendations on a major and the name of the school and nothing else.")

if os.path.exists(majors_file):
    # Create the assistant with modified instructions
    assistent = client.beta.assistants.create(
        model="gpt-3.5-turbo-1106",
        instructions= newInstructions,
        name="counclor",
        tools=[{"type": "retrieval"},{"type": "code_interpreter"}]
        
    )
else:
    print("Unable to create assistant. Majors file content is missing.")


#Make a thread for the assistant if the file is not missing
if majors_file:
    thread = client.beta.threads.create()
    pass
else:
    print("Unable to create assistant. Majors file content is missing.")




# Create the GUI
root = Tk()
root.title("Major Assistant")
root.configure(bg="black", padx=10, pady=10)
root.geometry("800x1000")

# Function to send the user's message to the assistant
def BUTTONpressed():
    if UserQueryTextBox.get(1.0,END).strip() == "":
        return 
    # Get the user's message and clear the text box and show it in the output box
    gptOutputTextBox.config(state=NORMAL)
    UserQuerySting = UserQueryTextBox.get(1.0,END).strip()
    
    gptOutputTextBox.insert(END,"\n\n\nUser: "+UserQuerySting)
    UserQueryTextBox.delete(1.0,END)
    
    # Send the user's message to the assistant
    my_thread_message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=UserQuerySting
    ) 
    # Get the assistant's response
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistent.id,
        instructions= newInstructions
    )
    # Get the status of the assistant's response and when it is completed, show the response in the output box
    while run.status in ["queued", "in_progress"]:
        getStatus = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        print(f"Run status: {getStatus.status}")  
        
        if getStatus.status == "completed":
            all_messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            gptOutputTextBox.insert(END, f"\n\n\nAssitant: {all_messages.data[0].content[0].text.value}")
            gptOutputTextBox.config(state=DISABLED)  
            
            break
        elif getStatus.status == "queued" or getStatus.status == "in_progress":
            pass
        else:
            print(f"Run status: {getStatus.status}")
            break    
        
    
    
#Gui elements 
titleLab = Label(root, text="Major Assistant", bg="black", fg="white", font=("Helvetica", 20))
titleLab.pack(pady=10,side=TOP)

gptOutputTextBox = Text(root, width=70, border=3, height=50, wrap=WORD, state=DISABLED)
gptOutputTextBox.pack(pady=10, side=TOP)
gptOutputTextBox.insert(END, "Welcome to the Major Assistant! How can I help you today?")

bottomFrame = Frame(root, bg="Grey", padx=2, pady=2, relief="raised", borderwidth=5,width=800,height=100)
bottomFrame.pack(side=BOTTOM)

mainButton = Button(root, text="Send Message",height=2,width=20,bg="grey",fg="white",command=BUTTONpressed)
mainButton.pack(in_=bottomFrame, side=RIGHT)

UserQueryTextBox = Text(root, width=70,border=3,height=5,wrap=WORD)
UserQueryTextBox.pack(in_=bottomFrame, side=LEFT,padx=5,pady=10)


root.mainloop()