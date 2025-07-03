from langchain_openai import AzureChatOpenAI
import json
try:
    from openai_resourse import get_openai_response
except ModuleNotFoundError:
    from src.openai_resourse import get_openai_response
 

output_schema = f""" 
    Summary : Front bumper detached with significant damage, grille missing, scratches on the bumper area, and potential impact damage to the front headlight and surrounding panels. 
    
    Cause of Damage : The potential cause of damage is a frontal collision with a pole. 
    
    Injuries : Possible injuries include head or chest injuries for the driver due to sudden impact, whiplash, or bruising from airbag deployment.
    """

delimiter = "###"
 
def severity_OpenAI_prediction(DamageDescription, ThirdpartyDescription):
    system_message_summary = f"""
    You are a helpful claims adjuster who is expert at analyzing the insurance claim description. You will be given below \
    the thirdparty damage description and damage description of the incident. They will be separated by a {delimiter}.
 
    1. First description will be the damage description
    2. Second description post {delimiter} will be third party description
 
    Use both these descriptions to write a coherent CASE SUMMARY for the claim. Keep in mind the \
    following guidelines :
    a. Use only the facts provided in the descriptions
    b. Include location details, damage caused and other keywords
    c. Your CASE SUMMARY should not exceed more than 200 words
    d. Incase of an irrelevant input, generate a message stating that the description given is irrelevant
 
    Your final answer is CASE SUMMARY. DO NOT include any other text.
    """
 
    system_message_category = f"""
 
    You are a helpful business analyst expert at identifying the severity of the insurance claim. You will be \
    given the raw text which will contain the description of the incident.
 
    Now using that description follow sequence of steps below:
 
    Step 1: {delimiter} Analyze the incident for number of parties involved
    Step 2: {delimiter} Determine if there was a major hospitalization or any fatalities or any \
    life threatening event was involved.
    Step 3: {delimiter} Understand keywords linked to bodily injuries if any
    Step 4: {delimiter} Classify the incident by selecting the most appropriate class from four classes\
        - fatal, severe, major, moderate, minor
 
    Your final output should be a single word which is the injury severity level which you decided in the fourth step based on the above description. Don’t give outputs for the rest of the steps.
    """
 
    system_message_category2 = f"""
    You are a helpful business analyst expert at identifying the **damage severity** of an insurance claim. YYou will be given below \
    the thirdparty damage description and damage description of the incident. They will be separated by a {delimiter}.
    
    1. First description will be the damage description
    2. Second description post {delimiter} will be third party description
 
    Using both these descriptions follow the sequence of steps below:
    
    If the given description is irrelevant return irrelevant. 
    If the given description has no damage return No Damage.
    
    Else follow the sequence of steps below:
 
    Step 1: {delimiter} Analyze the incident for the number of vehicles or properties involved
    Step 2: {delimiter} Determine if there was any total loss, structural damage, or major destruction reported
    Step 3: {delimiter} Understand keywords linked to types of damage like dent, scratch, replacement, frame damage, etc.
    Step 4: {delimiter} Classify the incident by selecting the most appropriate class from five classes\
        - Total Loss, Severe, Major, Moderate, Minor
    Your final output should be the single word which is the damage severity level which you decided in the fourth step based on the above description. Don’t give outputs for the rest of the steps.
"""
 
 
    messages_summary = [
        {"role": "system", "content": system_message_summary},
        {"role": "user", "content": f"{DamageDescription}{delimiter}{ThirdpartyDescription}"},
    ]
    summary = get_openai_response(messages_summary)
 
    try:
        messages_category = [
            {"role": "system", "content": system_message_category},
            {"role": "user", "content": f"{delimiter}{InjuryDescription}{delimiter}"},
        ]
        category = get_openai_response(messages_category)
    except:
        category = "fatal"
       
 
    messages_category2 = [
        {"role": "system", "content": system_message_category2},
        {"role": "user", "content": f"{CallDescription}"},
    ]
    category2 = get_openai_response(messages_category2)
     
    return {
        "summary": summary,
        "damage_severity": category2
    }


def details_OpenAI_prediction(CallTranscript):

    details_prompt = f"""You are an expert Claims Adjuster and you are given a call transcript. 
    Based on the call transcript given, you need to extract the following details from the call transcript : \
    Type of damage done to vehicle, the date of the incident, repair estimate, the location (including city and state or zip code), a detailed description of what happened, any injuries sustained,\
    the make and model of the vehicle (including the year), the estimated cost of damages. Generate a paragraph(summary) based on the extracted details. 
    Also, analyze the trancript for causes of damage and possible injuries sustained by the driver and the passengers. 

    Format the output according to the example given : {output_schema}     

    Incase of an irrelevant input, generate a message stating that the description given is irrelevant.
    Important Note:  Use only the facts provided in the descriptions"""

    damage_severity_prompt = f"""
    You are a helpful business analyst expert at identifying the **damage severity** of an insurance claim. You will be given below \
    the damage description of the incident.

    If the given description is irrelevant return irrelevant. 
    If the given description has no damage return No Damage.
    
    Else follow the sequence of steps below:
 
    Step 1: Analyze the incident for the number of vehicles or properties involved
    Step 2: Determine if there was any total loss, structural damage, or major destruction reported
    Step 3: Understand keywords linked to types of damage like dent, scratch, replacement, frame damage, etc.
    Step 4: Classify the incident by selecting the most appropriate class from five classes\
        - Total Loss, Severe, Major, Moderate, Minor
    Your final output should be the single word which is the damage severity level which you decided in the fourth step based on the above description. Don’t give outputs for the rest of the steps.
"""

    transcript_summary = [
        {"role": "system", "content": details_prompt},
        {"role": "user", "content": f"{CallTranscript}"},
    ]

    summary = get_openai_response(transcript_summary)

    damage_severity = [
        {"role": "system", "content": damage_severity_prompt},
        {"role": "user", "content": f"{summary}"},
    ]

    severity_level = get_openai_response(damage_severity)

    return {
        "summary": summary,
        "damage_severity": severity_level
    }
