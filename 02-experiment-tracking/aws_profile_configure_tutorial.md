# AWS profile setup

This tutorials explains how to configure one aws profile in your personal machine.

1. Check if AWS CLI is installed with the command `aws --version`. If not installed, run the command `pip install awscli` in the terminal.

2. Use the command `aws configure --profile my-profile-name` in the terminal. Answer the questions:

- AWS Access Key ID [None]: `your access key`
AWS Secret Access Key [None]: `your secret access key`
Default region name [None]: `press ENTER`
Default output format [None]: `press ENTER`

3. Once the profile exists, you can set it in a python script with:

`import os`

`os.environ["AWS_PROFILE"] = "my-profile-name"``

Once this is done, one can perform actions that have restricted authorization, given that the user is authorized to do them (example: store files in a S3 bucket from one's account)

