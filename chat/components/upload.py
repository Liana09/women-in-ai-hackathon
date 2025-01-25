def handle_upload(file):
    # Save the uploaded file to a desired location
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    print(f"File saved to {file_path}")
    # Perform any additional processing if needed
