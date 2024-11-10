import fitz  # PyMuPDF
import google.generativeai as genai
import json
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
from groq import Groq
import io
import re

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def verification():
    return "API is running"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'medical_bill' not in request.files or \
       'doctor_report' not in request.files or \
       'test_report' not in request.files or \
       'policy_path' not in request.files:
        return jsonify({"error": "Missing file(s)"}), 400

    files = {
        'medical_bill': request.files['medical_bill'],
        'doctor_report': request.files['doctor_report'],
        'test_report': request.files['test_report'],
        'policy_path': request.files['policy_path']
    }

    file_addresses = []
    for key, file in files.items():
        if file and allowed_file(file.filename):
            filename = file.filename
            saved_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_addresses.append(saved_file)
            file.save(saved_file)
        else:
            return jsonify({"error": f"File {key} is not a valid PDF"}), 400


    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    verifier = InsuranceDocumentVerifier(GEMINI_API_KEY)
    
    # Sample form data (use the JSON you provided)
    form_data = {
"insuranceType": "medicare",
"insuredId": "123",
"insuredName": "sam",
"insuredAddress": "Tower C-306 , Cosmos Golden Height",
"insuredCity": "Ghaziabad",
"insuredState": "Uttar Pradesh",
"insuredZip": "201016",
"insuredPhone": "07900507270",
"insuredDob": "2024-11-01",
"insuredSex": "M",
"insuredEmployer": "ram",
"insurancePlan": "",
"otherHealthPlan": False,
"patientName": "sam",
"patientAddress": "",
"patientCity": "",
"patientState": "",
"patientZip": "",
"patientPhone": "",
"patientDob": "2024-11-05",
"patientSex": "M",
"patientStatus": "single",
"patientEmployment": "employed",
"patientRelationship": "self",
"isEmploymentRelated": False,
"isAutoAccident": False,
"autoAccidentState": "",
"isOtherAccident": True,
"illnessDate": "",
"similarIllnessDate": "",
"unableToWorkFrom": "",
"unableToWorkTo": "",
"referringPhysician": "",
"referringPhysicianId": "",
"hospitalizationFrom": "",
"hospitalizationTo": "",
"diagnosis1": "D1",
"diagnosis2": "D2",
"diagnosis3": "",
"diagnosis4": "",
"outsideLab": False,
"outsideLabCharges": "",
"medicaidResubmissionCode": "",
"medicaidOriginalRef": "",
"priorAuthNumber": "",
"acceptAssignment": False,
"totalCharge": "40",
"amountPaid": "20",
"balanceDue": "20",
"federalTaxId": "",
"patientAccountNo": "22222222",
"providerName": "Bank",
"providerAddress": "Tower C-306 , Cosmos Golden Height",
"providerCity": "Ghaziabad",
"providerState": "Uttar Pradesh",
"providerZip": "",
"providerPhone": "07900507270"
}
    # Verify documents
    result = verifier.verify_documents(
        form_data,
        file_addresses[0],
        file_addresses[1],
        file_addresses[2],
        file_addresses[3]
    )
    
    # Print results
    print("\nDocument Verification Results:")
    print(f"Final Score: {result['final_score']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print("\nIndividual Document Scores:")
    for doc_type, score in result['individual_scores'].items():
        print(f"{doc_type}: {score}%")
        
    return jsonify({"message": "Files successfully uploaded",
                    "result": result
                    }), 200

    
# MODEL
class InsuranceDocumentVerifier:
    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        genai.configure(api_key=self.api_key)
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.system_prompts = {
            'medical_bill': """Analyze medical bill for inconsistencies and fraud indicators. Focus on:
1. Charges and service dates
2. Billing codes
3. Match with form data

Return JSON format:
{
    "confidence_score": "XX",
    "findings": "key findings in brief"
}""",

            'doctor_report': """Analyze doctor's report authenticity. Check:
1. Medical consistency
2. Timeline alignment
3. Treatment appropriateness

Return JSON format:
{
    "confidence_score": "XX",
    "findings": "key findings in brief"
}""",

            'test_report': """Verify lab test report authenticity. Check:
1. Result consistency
2. Timeline alignment
3. Format standards

Return JSON format:
{
    "confidence_score": "XX",
    "findings": "key findings in brief"
}""",

            'policy': """Verify insurance policy validity. Check:
1. Coverage details
2. Dates and terms
3. Document authenticity

Return JSON format:
{
    "confidence_score": "XX",
    "findings": "key findings in brief"
}"""
        }

    def extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            for page_num in range(min(2, doc.page_count)):
                text_content += doc[page_num].get_text()
            doc.close()
            return text_content
        except Exception as e:
            print(f"Error extracting PDF content: {str(e)}")
            return ""

    def analyze_document(self, doc_type: str, text_content: str, form_data: Dict) -> Dict[str, Any]:
        """Analyze document using Groq"""
        try:
            essential_fields = ['insuredName', 'patientName', 'totalCharge', 'diagnosis1']
            filtered_form_data = {k: form_data[k] for k in essential_fields if k in form_data}
            
            prompt = f"""{self.system_prompts[doc_type]}

                        Key Document Text:
                        {text_content[:500]}

                        Essential Form Data:
                        {json.dumps(filtered_form_data)}

                        Respond only in the specified JSON format."""

            client = Groq()
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                stream=False,
                max_tokens=150
            )

            # Attempt to extract JSON from response
            response_text = chat_completion.choices[0].message.content
            # Find JSON pattern in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"confidence_score": "50", "findings": "Error parsing response"}
            else:
                return {"confidence_score": "50", "findings": "No structured response found"}

        except Exception as e:
            print(f"Error in analyze_document: {str(e)}")
            return {"confidence_score": "50", "findings": f"Analysis error: {str(e)}"}

    def cross_validate_documents(self, analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Cross-validate information between different documents"""
        try:
            cross_validation_prompt = f"""Analyze these document analyses for consistency:

{json.dumps(analyses, indent=2)}

Return only in this JSON format:
{{
    "consistency_score": "XX",
    "recommendation": "Approve/Review/Reject",
    "reasoning": "brief key finding"
}}"""

            self.analyse_model = genai.GenerativeModel('gemini-1.5-pro')
            response = self.analyse_model.generate_content(cross_validation_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {
                        "consistency_score": "50",
                        "recommendation": "Review",
                        "reasoning": "Error parsing cross-validation response"
                    }
            else:
                return {
                    "consistency_score": "50",
                    "recommendation": "Review",
                    "reasoning": "No structured response found"
                }

        except Exception as e:
            print(f"Error in cross_validate_documents: {str(e)}")
            return {
                "consistency_score": "50",
                "recommendation": "Review",
                "reasoning": f"Cross-validation error: {str(e)}"
            }

    def calculate_final_score(self, analyses: Dict, cross_validation: Dict) -> Dict[str, Any]:
        """Calculate final verification score and compile report"""
        try:
            scores = {}
            for doc_type, analysis in analyses.items():
                try:
                    score = float(analysis.get('confidence_score', '50'))
                    scores[doc_type] = score
                except (ValueError, AttributeError):
                    scores[doc_type] = 50.0

            try:
                cross_val_score = float(cross_validation.get('consistency_score', '50'))
            except (ValueError, AttributeError):
                cross_val_score = 50.0

            scores['cross_validation'] = cross_val_score
            
            weights = {
                'medical_bill': 0.25,
                'doctor_report': 0.2,
                'test_report': 0.2,
                'policy': 0.15,
                'cross_validation': 0.2
            }
            
            final_score = sum(scores[k] * weights[k] for k in weights)
            
            return {
                'final_score': round(final_score, 2),
                'risk_level': "Low" if final_score >= 80 else "Medium" if final_score >= 60 else "High",
                'individual_scores': scores,
                'recommendation': cross_validation.get('recommendation', 'Review'),
                'key_finding': cross_validation.get('reasoning', 'No specific findings')
            }

        except Exception as e:
            print(f"Error in calculate_final_score: {str(e)}")
            return {
                'final_score': 50.0,
                'risk_level': "High",
                'individual_scores': {doc_type: 50.0 for doc_type in analyses.keys()},
                'recommendation': "Review",
                'key_finding': f"Error in score calculation: {str(e)}"
            }

    def verify_documents(self, form_data: Dict, 
                         medical_bill_path: str,
                        doctor_report_path: str, 
                        test_report_path: str,
                        policy_path: str) -> Dict[str, Any]:
        """Main function to verify all documents"""
        documents = {
            'medical_bill': medical_bill_path,
            'doctor_report': doctor_report_path,
            'test_report': test_report_path,
            'policy': policy_path
        }
        
        analyses = {}
        for doc_type, path in documents.items():
            text_content = self.extract_pdf_content(path)
            analyses[doc_type] = self.analyze_document(doc_type, text_content, form_data)
        
        cross_validation = self.cross_validate_documents(analyses)
        return self.calculate_final_score(analyses, cross_validation)

app.run(debug=False)