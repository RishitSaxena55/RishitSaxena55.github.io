import fitz

doc = fitz.open('Rishit_Saxena_CV (2).pdf')
links = [link['uri'] for page in doc for link in page.get_links() if 'uri' in link]

with open('all_links.txt', 'w', encoding='utf-8') as f:
    for i, l in enumerate(links):
        f.write(f'{i+1}. {l}\n')
    f.write(f'\nTotal: {len(links)} links\n')
print('Done!')
