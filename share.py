from flask import Flask, request
from Crypto.Cipher import AES
from crypto
import base64

app = Flask(__name__)

# Secret key for encryption
secret_key = "mysecretkey"

# Function to encrypt password
def encrypt_password(password):
    cipher = AES.new(secret_key, AES.MODE_ECB)
    return base64.b64encode(cipher.encrypt(password))

# Function to decrypt password
def decrypt_password(encrypted_password):
    cipher = AES.new(secret_key, AES.MODE_ECB)
    return cipher.decrypt(base64.b64decode(encrypted_password)).strip()

# API endpoint to share password
@app.route('/share_password', methods=['POST'])
def share_password():
    if request.method == 'POST':
        # Get the password from the request
        password = request.form['password']
        # Encrypt the password
        encrypted_password = encrypt_password(password)
        # Share the encrypted password
        return encrypted_password

# API endpoint to retrieve shared password
@app.route('/get_password', methods=['GET'])
def get_password():
    if request.method == 'GET':
        # Get the encrypted password from the request
        encrypted_password = request.args.get('encrypted_password')
        # Decrypt the password
        password = decrypt_password(encrypted_password)
        # Return the decrypted password
        return password

if __name__ == '__main__':
    app.run(debug=True)
