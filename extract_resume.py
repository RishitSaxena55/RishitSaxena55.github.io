import fitz

doc = fitz.open('Rishit_Saxena_CV (2).pdf')
with open('resume_content.md', 'w', encoding='utf-8') as f:
    for i, page in enumerate(doc):
        f.write(f'## PAGE {i+1}\n\n')
        f.write(page.get_text())
        f.write('\n\n')
print("Resume content extracted to resume_content.md")
