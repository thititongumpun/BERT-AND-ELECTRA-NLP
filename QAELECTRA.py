from transformers import pipeline
import streamlit as st

model_name = "deepset/electra-base-squad2"
electra = pipeline('question-answering',
  model=model_name,
  tokenizer=model_name
)

class generateQA:
  def __init__(self, question, context):
    self.question = question
    self.context = context
qa = []
qa.append(generateQA('What is my name?', 'My name is Thiti Tongumpun. Im 27  year old. I have no life!'))
qa.append(generateQA('Why is model conversion important?',
        'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'))
qa.append(generateQA('The New York Giants and the New York Jets play at which stadium in NYC?',
        'The city is represented in National Football League by the New York Giants and the New York Jets, \
         although both teams play thier home games at MetLife Stadium in nearby East Rutherford, New Jersey, which, hosted Super Bowl XLVIII in 2014.'))
qa.append(generateQA('When did Beyonce start becoming popular?', 'in the late 1990s' ))

q1 = qa[0]
q2 = qa[1]
q3 = qa[2]
q4 = qa[3]

def init():
  st.write('# Question-Answering')

  selected = st.selectbox('Question', (
    qa[0].question,
    qa[1].question,
    qa[2].question,
    qa[3].question,
  ))
  
  st.write('Context')
  if (selected == qa[0].question):
    st.write(qa[0].context)
  elif (selected == qa[1].question):
    st.write(qa[1].context)
  elif (selected == qa[2].question):
    st.write(qa[2].context)
  elif (selected == qa[3].question):
    st.write(qa[3].context)

  if (selected == qa[0].question):
    q = {
      'question': q1.question,
      'context': q1.context
    }
    result = electra(q)
  elif (selected == qa[1].question):
    q = {
      'question': q2.question,
      'context': q2.context
    }
    result = electra(q)
  elif (selected == qa[2].question):
    q = {
      'question': q3.question,
      'context': q3.context
    }
    result = electra(q)
  elif (selected == qa[3].question):
    q = {
      'question': q4.question,
      'context': q4.context
    }
    result = electra(q)

  st.write(result)

if __name__ == '__main__':
  init()
