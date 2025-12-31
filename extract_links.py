import fitz

doc = fitz.open('Rishit_Saxena_CV (2).pdf')
print('=== ALL EMBEDDED LINKS IN RESUME ===\n')
for i, page in enumerate(doc):
    links = page.get_links()
    for link in links:
        if 'uri' in link:
            print(link['uri'])
