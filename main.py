# import Library which are necessary
import streamlit as st
from io import StringIO         # Taking text data from txt file
from bertopic import BERTopic   #Iopic modeling

# Header for our streamlit page
st.header("Welcome")

# Below code create a sidebar of the page and display the following content
with st.sidebar:
    st.header("What is Topic Modeling?")
    st.write("Topic models can help to organize and offer insights for us to understand large collections of unstructured text bodies. Originally developed as a text-mining tool, topic models have been used to detect instructive structures in data such as genetic information, images, and networks.")

# loading the model 
model = BERTopic.load("topic_model", embedding_model="all-mpnet-base-v2")

# this function will take the text data and perform topic modeling
def compute(text):
    labels = []
    topics, probs = model.transform(text)
    st.write("-----------------------------------------------------------------------------------")
    st.write("Predicted Topic id: ",topics)
    topic_name = model.topic_names[topics[0]]
    st.write("Predicted Topic name: ",topic_name)
    st.write("-----------------------------------------------------------------------------------")
    st.write("Suggested Topic : \n")
    result = model.find_topics(text)
    for index in result[0]:
        st.write(model.topic_names[index],"\n")
    st.write("-----------------------------------------------------------------------------------")
    # For displaying output
    for index1 in result[0]:
        temp = model.get_topic(index1) 
        st.write(temp)
        for index2 in temp:
            labels.append(index2)
        st.write("-----------------------------------------------------------------------------------")
        st.write(model.visualize_topics())
        st.write(model.visualize_barchart())
        st.write(model.visualize_heatmap())
        st.write(model.visualize_hierarchy())
        st.write(model.visualize_term_rank())
    return(labels)


# below code will create a drop box for uploading the file
uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)


# we will only take .txt file as of now and then take txt file and take data in string data
if uploaded_file is not None:
    if uploaded_file.name.split(".")[1] != "txt":
        st.write("Please Upload txt file!!!!")
    # this part will only excute if file is .txt
    else:
        st.write("This might take few seconds")
        st.header("Reading File Content")
        # taking the text data from file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.write(string_data)
        labels = compute(string_data)