import json
import boto3
import mailparser
import string
import sys
from hashlib import md5
import numpy as np
from io import StringIO
import os
import csv
from botocore.exceptions import ClientError


s3_client = boto3.client("s3")
S3_BUCKET = 'a3-s1-test'
vocabulary_length = 9013
ENDPOINT_NAME = "sms-spam-classifier-mxnet-2022-11-22-16-40-14-473"
runtime= boto3.client('runtime.sagemaker')

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
#
#
# these functions are copied from 
# https://github.com/aws-samples/reinvent2018-srv404-lambda-sagemaker/blob/master/training/sms_spam_classifier_utilities.py
#
#

# test comment

def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

#
# 
# Send email code from https://github.com/RekhuGopal/PythonHacks/blob/main/AWSBoto3Hacks/AWSBoto3-SES-Lambda.py
#
#

def send_email(EMAIL_RECIPIENT, EMAIL_RECEIVE_DATE, EMAIL_SUBJECT, EMAIL_BODY, CLASSIFICATION, CLASSIFICATION_CONFIDENCE_SCORE):
    SENDER = "ccbd-a32022@hikarujorden.com" # must be verified in AWS SES Email
    RECIPIENT = EMAIL_RECIPIENT # must be verified in AWS SES Email

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"

    # The subject line for the email.
    SUBJECT = "Email Received: Classification Email"
    
    BODY_TEXT = '''We received your email sent at {} with the subject {}.
    Here is a 240 character sample of the email body: {} 
    The email was categorized as {} with a 
    {}% confidence.
    '''.format(EMAIL_RECEIVE_DATE, EMAIL_SUBJECT, EMAIL_BODY, CLASSIFICATION, CLASSIFICATION_CONFIDENCE_SCORE)
    
    BODY_HTML = '''<html>\
    <head></head>
    <body>
    <p><b>We received your email sent at {} with the subject {}. 
    Here is a 240 character sample of the email body: </b>{} 
    <b>The email was categorized as {} with a
    {}% confidence.</
    </p>
    </body>
    </html>\
    '''.format(EMAIL_RECEIVE_DATE, EMAIL_SUBJECT, EMAIL_BODY, CLASSIFICATION, CLASSIFICATION_CONFIDENCE_SCORE)

    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)

    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
        
                        'Data': BODY_HTML
                    },
                    'Text': {
        
                        'Data': BODY_TEXT
                    },
                },
                'Subject': {

                    'Data': SUBJECT
                },
            },
            Source=SENDER
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])


def lambda_handler(event, context):
    
    object_key = event["Records"][0]["s3"]["object"]["key"]
    file_content = s3_client.get_object(
    Bucket=S3_BUCKET, Key=object_key)["Body"].read()
    print('file_content: ',file_content)
    
    mail = mailparser.parse_from_bytes(file_content)
    
    #according to edstem https://edstem.org/us/courses/29541/discussion/2195609
    #"use the text_plain attribute"
    email_body = mail.text_plain #this is a list object
    email_body[0] = email_body[0].replace("\r\n", " ") 
    #print(email_body) 
    
    #according to edstem: 
    #Refer to how one_hot_encode and vectorize_sequences have been used in the training notebook. 
    #You are supposed to encode the body of the email in a similar way. 
    
    # one hot encoding for each SMS message
    one_hot_data = one_hot_encode(email_body, vocabulary_length)
    encoded_messages = vectorize_sequences(one_hot_data, vocabulary_length)
    
    #Once encoded, do the following before sending data to the prediction endpoint:
    io = StringIO()
    json.dump(encoded_messages.tolist(), io)
    body = bytes(io.getvalue(), 'utf-8')
    
    #get the CLASSIFICATION and CLASSIFICATION_CONFIDENCE_SCORE
    
    #https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=body)
    #print(response)
    result = json.loads(response['Body'].read().decode())
    #print(result)
    pred = int(result['predicted_label'][0][0])
    #print(pred)
    classification = 'SPAM' if pred == 1 else 'HAM'
    confidence = round(result['predicted_probability'][0][0]*100)

    #body for email must be 240 characters.
    email_body = (email_body[:240]) if len(email_body) > 240 else email_body
    # send the email
    send_email(mail.from_[0][1], mail.date, mail.subject, email_body, classification, confidence)

    
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
