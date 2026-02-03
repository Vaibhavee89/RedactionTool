from reportlab.pdfgen import canvas

def create_pdf(filename):
    c = canvas.Canvas(filename)
    c.drawString(100, 750, "Confidential Document")
    c.drawString(100, 730, "Name: John Doe")
    c.drawString(100, 710, "Email: john.doe@example.com")
    c.drawString(100, 690, "Phone: 9876543210")
    c.drawString(100, 670, "PAN: ABCDE1234F")
    c.save()

if __name__ == "__main__":
    create_pdf("sample.pdf")
