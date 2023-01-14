import streamlit as st
from streamlit.web import cli as stcli
from streamlit import runtime
import sys


st.header("Welcome to my Streamlit Website")

st.markdown("Here is some **bold text** and *italicized text*.")

st.sidebar.header("About")
st.sidebar.markdown("This is a simple Streamlit website.")

name = st.text_input("Enter your name:")
age = st.slider("Enter your age:", min_value=0, max_value=120, value=20)

if st.button("Submit"):
    st.write("Your name is", name, "and your age is", age)

if __name__ == '__main__':
    if runtime.exists():
        print("")
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())