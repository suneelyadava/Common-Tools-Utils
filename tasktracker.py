import streamlit as st
from streamlit.web import cli as stcli
from streamlit import runtime
import sys
import streamlit as st
# Set the title and header of the page
st.title("User Login")
st.header("Log in with Streamlit")

# Create a dictionary to store users and their details
users = {}

# Load the user data from the text file
with open("users.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        username, email, password = line.strip().split(",")
        users[username] = {"email": email, "password": password}

# Create the login and signup tabs
tab = st.sidebar.radio("", ["Login", "Sign Up"])

# Display the login form
if tab == "Login":
    st.header("Login Form")
    username = st.text_input("Enter your username:")
    password = st.text_input("Enter your password:", type="password")
    if st.button("Log In"):
        if username in users and users[username]["password"] == password:
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password")

# Display the signup form
if tab == "Sign Up":
    st.header("Sign Up Form")
    new_username = st.text_input("Enter a new username:")
    new_email = st.text_input("Enter your email:")
    new_password = st.text_input("Enter a new password:", type="password")
    if st.button("Sign Up"):
        if new_username in users:
            st.error("Username already taken")
        else:
            users[new_username] = {"email": new_email, "password": new_password}
            st.success("Successfully registered!")

# Save the user data to the text file
with open("users.txt", "w") as f:
    for username, user in users.items():
        f.write(username + "," + user["email"] + "," + user["password"] + "\n")

if __name__ == '__main__':
    if runtime.exists():
        print("")
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
