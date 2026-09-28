"""原論文の図・別行数式を維持し、日本語訳を原位置に組版する。"""
from pathlib import Path
import argparse
import json
import re
import pymupdf as fitz

EXPERIMENT_KIND = 'paper_translation'
SOURCE = Path(r'C:\Users\hikar\Downloads\2103.05629v2.pdf')
RUN = Path('results/2026-09-28/paper_translation/v1_2103_05629_ja')


def read_regions():
    text = (RUN / 'translation.txt').read_text(encoding='utf-8')
    regions = []
    for piece in re.split(r'^@', text, flags=re.M)[1:]:
        head, body = piece.split('\n', 1)
        args = head.split('|')
        page = int(args[0])
        if args[1] in ['L', 'R', 'F']:
            x0, x1 = {'L': (53.8, 299.4), 'R': (316.8, 562.5), 'F': (53.8, 562.5)}[args[1]]
            rect = [x0, float(args[2]), x1, float(args[3])]
            size = float(args[4]) if len(args) > 4 else 9.2
            align = args[5] if len(args) > 5 else 'left'
        else:
            rect = list(map(float, args[1].split(',')))
            size = float(args[2])
            align = args[3]
        regions.append(dict(page=page, rect=rect, text=body.strip(), size=size, align=align))
    return regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='draft')
    parser.add_argument('--render', default='1,2,3')
    parser.add_argument('--output-name', default='translated_ja.pdf')
    args = parser.parse_args()
    target = RUN / (args.output_name if args.tag.startswith('final') else f'qa/{args.tag}.pdf')
    if target.exists():
        raise FileExistsError(target)
    regions = read_regions()
    doc = fitz.open(SOURCE)
    archive = fitz.Archive(r'C:\Windows\Fonts')
    css_base = '''
    @font-face {font-family: JP; src: url(yumin.ttf);}
    @font-face {font-family: JP; src: url(yumindb.ttf); font-weight: bold;}
    * { margin: 0; padding: 0; }
    body {font-family: JP; color: #000; line-height: 1.28;}
    sup, sub {font-size: 70%;}
    p {margin:0 0 0.4em 0;}
    '''
    report = []
    for n, page in enumerate(doc, 1):
        entries = [r for r in regions if r['page'] == n]
        for r in entries:
            page.add_redact_annot(fitz.Rect(r['rect']), fill=(1, 1, 1), cross_out=False)
        if entries:
            page.apply_redactions(images=0, graphics=0)
        for i, r in enumerate(entries):
            rect = fitz.Rect(r['rect'])
            css = css_base + f"body {{font-size:{r['size']}pt;text-align:{r['align']};}}"
            spare, scale = page.insert_htmlbox(rect, r['text'], css=css, archive=archive, scale_low=0.65)
            report.append({**r, 'index': i, 'spare': spare, 'scale': scale, 'actual_fontsize': r['size'] * scale})
            if spare < 0:
                raise RuntimeError(f'Text overflow page {n}, region {i}')
    doc.set_metadata({'title': 'コヒーレントイジングマシンによる基底・低エネルギーイジングスピン配置の効率的サンプリング（日本語訳）',
                      'author': 'Edwin Ng et al.',
                      'subject': 'arXiv:2103.05629v2 日本語訳。図と別行数式は原文を保持。',
                      'keywords': 'MFB-CIM, coherent Ising machine, Gaussian state, Japanese translation'})
    doc.set_toc([
        [1, 'I. はじめに', 1],
        [1, 'II. MFB-CIMの離散時間ガウス量子モデル', 2],
        [2, 'A. 基本形式', 4], [2, 'B. 非線形結晶内の伝搬', 6],
        [2, 'C. 離散時間動力学モデル', 7], [2, 'D. 連続時間ガウスモデルへの帰着', 8],
        [1, 'III. 数値結果', 9], [2, 'A. モデルのパラメータ', 9],
        [2, 'B. イジングサンプリング', 11], [2, 'C. 代替モデルでのサンプリング', 13],
        [1, 'IV. サンプリング性能のスケーリング推定', 15],
        [1, 'V. 結論', 18], [1, '謝辞', 18], [1, '参考文献', 19],
        [1, '付録A：連続時間ガウス量子モデルと高フィネス極限', 20],
        [2, '1. 連続時間ガウス量子モデル', 20], [2, '2. 高フィネス極限', 21],
        [2, '3. 量子入出力理論による方法', 22],
        [1, '付録B：直交位相演算子の期待値の評価', 23],
        [1, '付録C：結晶内伝搬の運動方程式', 23],
    ])
    doc.subset_fonts()
    doc.save(target, garbage=4, deflate=True)
    (RUN / f'qa/{args.tag}_layout.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SAVED', target.resolve(), 'PAGES', len(doc), 'REGIONS', len(regions))
    print('SMALLEST', [(r['page'], r['index'], round(r['actual_fontsize'], 2)) for r in sorted(report, key=lambda r:r['actual_fontsize'])[:12]])
    for n in [int(v) for v in args.render.split(',') if v]:
        doc[n-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(RUN / f'qa/{args.tag}_{n:02}.png')


if __name__ == '__main__':
    main()
