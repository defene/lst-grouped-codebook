"""Package verified scientific PNG plates as two lossless, bookmarked PDF atlases."""
from pathlib import Path
import hashlib
import json

from PIL import Image
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'runs/paper1/codebook1024/full/report/more_predictions'
OUTPUT = ROOT/'output/pdf'


def main(source=SOURCE, prefix='VQVAE', subject='4096 vs 1024'):
    SOURCE = Path(source)
    manifest = json.loads((SOURCE/'figure_manifest.json').read_text(encoding='utf-8'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for modality in ('hls', 'lst'):
        pages = [r for r in manifest['pages'] if r['modality']==modality]
        assert len(pages) == 16 and [r['sample_number'] for r in pages]==list(range(1, 17))
        output = OUTPUT/f'{prefix}_{modality.upper()}_16_samples.pdf'
        doc = canvas.Canvas(str(output), pageCompression=1)
        doc.setTitle(f'{prefix} {modality.upper()} - 16 validation samples - {subject}')
        doc.setAuthor('EO reconstruction experiment')
        doc.setSubject('Frozen checkpoints; clean and masked inputs; shared-scale detail comparisons')
        for row in pages:
            image = SOURCE/row['image']
            assert hashlib.sha256(image.read_bytes()).hexdigest()==row['sha256']
            with Image.open(image) as im:
                width, height = [v*72/160 for v in im.size]
            doc.setPageSize((width, height))
            key = f"sample_{row['sample_number']:02d}"
            doc.bookmarkPage(key)
            doc.addOutlineEntry(f"{row['sample_number']:02d} | prithvi:val:{row['record_id']}", key)
            doc.drawImage(ImageReader(str(image)), 0, 0, width=width, height=height)
            doc.showPage()
        doc.save()
        pdf = PdfReader(output)
        assert len(pdf.pages)==16 and len(pdf.outline)==16
        outputs.append(dict(modality=modality, path=str(output.relative_to(ROOT)), pages=16,
                            bytes=output.stat().st_size, sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
        print(json.dumps(outputs[-1]), flush=True)
    (SOURCE/'pdf_manifest.json').write_text(json.dumps(outputs, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
