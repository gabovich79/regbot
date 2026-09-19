"""Stage original-backed versions with one shared batch budget; never activate them."""
import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


async def run(args):
    # Set the target before config/model imports; production remains untouched.
    root=Path(args.data_dir).resolve()
    if not (root/'regbot.db').is_file():
        raise ValueError('Existing copied database required')
    if not math.isfinite(args.budget) or args.budget<=0:
        raise ValueError('A finite positive batch budget is required')
    os.environ['DATA_DIR']=str(root)
    from models.database import init_db,get_db
    from services.providers import Gateway,BudgetExceeded
    from services.knowledge import stage_document
    from services.document_service import extract_pdf_pages,extract_docx
    await init_db()
    gateway=Gateway(purpose='indexing',limit=args.budget)
    report={'data_dir':str(root),'budget_usd':args.budget,'documents':[],'activated':False}
    db=await get_db()
    try:
        for document_id in dict.fromkeys(args.documents):
            try:
                row=await (await db.execute('SELECT * FROM documents WHERE id=? AND is_active=1',(document_id,))).fetchone()
                if not row:
                    raise ValueError('Active document not found')
                document=dict(row)
                path=Path(document.get('original_path') or '')
                if not path.is_file() or not path.resolve().is_relative_to(root):
                    raise ValueError('Original must exist within the copied data directory')
                pages=None
                if path.suffix.lower()=='.pdf':
                    pages=extract_pdf_pages(str(path));text=''
                elif path.suffix.lower()=='.docx':
                    text=extract_docx(str(path))
                else:
                    raise ValueError('Unsupported original: convert to PDF/DOCX first')
                version,count=await stage_document(db,document,text,pages,gateway=gateway)
                report['documents'].append({'id':document_id,'status':'staged','version':version,'chunks':count})
            except BudgetExceeded as exc:
                report['documents'].append({'id':document_id,'status':'budget_stopped','error':str(exc)})
                break
            except Exception as exc:
                report['documents'].append({'id':document_id,'status':'failed','error':str(exc)})
            finally:
                report['cost_usd']=gateway.spent
                Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        await db.close()
    print(json.dumps(report,ensure_ascii=False))
    return all(d['status']=='staged' for d in report['documents']) and len(report['documents'])==len(set(args.documents))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--documents',type=int,nargs='+',required=True)
    parser.add_argument('--budget',type=float,required=True)
    parser.add_argument('--output',required=True)
    sys.exit(0 if asyncio.run(run(parser.parse_args())) else 1)
