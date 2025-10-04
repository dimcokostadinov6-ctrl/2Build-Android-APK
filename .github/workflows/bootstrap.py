#!/usr/bin/env python3
# Генерира целия проект (код + buildozer.spec) в работната директория.
from pathlib import Path

def w(path: str, content: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

# ---------- core ----------
w("core/__init__.py", "")
w("core/ports.py", """from typing import List, Tuple, Protocol, Optional
import numpy as np

class IHandwriteAI(Protocol):
    def recognize_names(self, page_bgr: 'np.ndarray', known_names: List[str], min_conf: float = 0.6) -> List[str]: ...

class IOCR(Protocol):
    def parse_entries(self, image_path: str, user_words: Optional[List[str]] = None) -> List[Tuple[str,int]]: ...

class IRepository(Protocol):
    def init(self) -> None: ...
    def add_page(self, path: str, ts: str) -> int: ...
    def add_entry(self, name: str, amount_st: int, ts: str, page_id: int | None) -> None: ...
    def all_known_names(self) -> List[str]: ...
    def search_by_name(self, q: str) -> List[Tuple[str,int]]: ...
""")

w("core/services.py", """from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .ports import IHandwriteAI, IOCR, IRepository
from infra.ai.recognizer import mask_crossed_out_names

@dataclass
class SavePageService:
    repo: IRepository
    ai: IHandwriteAI
    ocr: IOCR

    def save_drawn_page(self, image_path: str, ts_iso: str) -> tuple[int, int]:
        self.last_removed_count = 0
        page_id = self.repo.add_page(image_path, ts_iso)

        try: known = self.repo.all_known_names()
        except Exception: known = []

        # 1) направи маска за задрасканите имена (един дълъг хор. щрих напред-назад)
        try:
            import cv2, os
            page_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if page_bgr is None:
                masked_path = image_path
            else:
                masked_bgr, removed = mask_crossed_out_names(page_bgr, left_ratio=0.6)
                self.last_removed_count = len(removed)
                base, ext = os.path.splitext(image_path)
                masked_path = base + "_masked" + (ext if ext else ".png")
                cv2.imwrite(masked_path, masked_bgr)
        except Exception:
            masked_path = image_path

        # 2) OCR извличане на (име, сума)
        try:
            ai_names: List[str] = []
            user_words = list({*(known or []), *ai_names})
            entries = self.ocr.parse_entries(masked_path, user_words=user_words)
        except Exception:
            entries = []

        # 3) запис в БД
        n = 0
        for name, amount_st in entries:
            if not name or amount_st is None: continue
            self.repo.add_entry(name=name.strip(), amount_st=int(amount_st), ts=ts_iso, page_id=page_id)
            n += 1
        return page_id, n
""")

# ---------- infra ----------
w("infra/__init__.py", "")
w("infra/database_sqlite.py", """import sqlite3
from pathlib import Path
from typing import List, Tuple
from core.ports import IRepository

DB_PATH = Path(__file__).resolve().parent.parent / 'veresia.db'

class SQLiteRepo(IRepository):
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    def init(self) -> None:
        with self._conn() as con:
            cur = con.cursor()
            cur.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS pages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                ts TEXT NOT NULL
            );\"\"\")
            cur.execute(\"\"\"
            CREATE TABLE IF NOT EXISTS entries(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                amount_st INTEGER NOT NULL,
                ts TEXT NOT NULL,
                page_id INTEGER,
                FOREIGN KEY(page_id) REFERENCES pages(id)
            );\"\"\")
            con.commit()

    def add_page(self, path: str, ts: str) -> int:
        with self._conn() as con:
            cur = con.cursor()
            cur.execute('INSERT INTO pages(path, ts) VALUES(?,?)', (path, ts))
            con.commit()
            return cur.lastrowid

    def add_entry(self, name: str, amount_st: int, ts: str, page_id: int | None) -> None:
        with self._conn() as con:
            cur = con.cursor()
            cur.execute('INSERT INTO entries(name, amount_st, ts, page_id) VALUES(?,?,?,?)',
                        (name, int(amount_st), ts, page_id))
            con.commit()

    def all_known_names(self) -> List[str]:
        with self._conn() as con:
            cur = con.cursor()
            cur.execute('SELECT DISTINCT name FROM entries ORDER BY name COLLATE NOCASE')
            return [r[0] for r in cur.fetchall()]

    def search_by_name(self, q: str) -> List[Tuple[str,int]]:
        q = (q or '').strip()
        with self._conn() as con:
            cur = con.cursor()
            if q:
                cur.execute(\"\"\"
                    SELECT name, SUM(amount_st) as total
                    FROM entries
                    WHERE name LIKE ?
                    GROUP BY name
                    ORDER BY total DESC
                \"\"\", (f\"%{q}%\",))
            else:
                cur.execute(\"\"\"
                    SELECT name, SUM(amount_st) as total
                    FROM entries
                    GROUP BY name
                    ORDER BY total DESC
                \"\"\" )
            return [(r[0], int(r[1] or 0)) for r in cur.fetchall()]
""")

# ---------- OCR ----------
w("infra/ocr_bul.py", """import re
from typing import List, Tuple, Optional, Union
from pathlib import Path
from core.ports import IOCR

class TesseractOCR(IOCR):
    def _try_pytesseract_on_image(self, image_path: Union[str, Path]) -> str:
        try:
            import cv2, pytesseract
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None: return ''
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return pytesseract.image_to_string(gray, lang='bul+eng') or ''
        except Exception:
            return ''

    def _extract_lines(self, text: str) -> list:
        return [ln.strip() for ln in (text or '').splitlines() if ln.strip()]

    def _parse_amount(self, s: str) -> Optional[int]:
        nums = re.findall(r'([0-9]+(?:[\\.,][0-9]{1,2})?)', s.replace(' ', ''))
        if not nums: return None
        val = nums[-1].replace(',', '.')
        try: return int(round(float(val)*100))
        except Exception: return None

    def _name_part(self, s: str) -> str:
        import re as _re
        parts = _re.split(r'([0-9]+(?:[\\.,][0-9]{1,2})?)', s)
        if not parts: return ''
        return (''.join(parts[:-2]).strip(' -:/\\t') if len(parts) >= 3 else s.strip())

    def parse_entries(self, image_path: str, user_words: Optional[list] = None) -> List[Tuple[str, int]]:
        text = self._try_pytesseract_on_image(image_path)
        if not text: return []
        out=[]
        for ln in self._extract_lines(text):
            amt = self._parse_amount(ln)
            if amt is None: continue
            nm = self._name_part(ln).strip()
            if nm: out.append((nm, amt))
        return out
""")

# ---------- AI (детекция на думи + задраскване) ----------
w("infra/ai/__init__.py", "")
w("infra/ai/recognizer.py", """from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np, cv2, unicodedata

AI_DIR = Path(__file__).resolve().parent
MODEL_PATH = AI_DIR / "ai_model" / "prototypes.npz"  # оставено за бъдещи разширения

_LAT_TO_CYR = str.maketrans({"A":"А","a":"а","B":"В","E":"Е","e":"е","K":"К","k":"к","M":"М","m":"м","H":"Н","O":"О","o":"о","P":"Р","p":"р","C":"С","c":"с","T":"Т","t":"т","X":"Х","x":"х","Y":"У","y":"у"})
def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").strip()).translate(_LAT_TO_CYR).casefold()
    return s

def _binarize(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3,3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    return cv2.medianBlur(bw,3)

def extract_word_boxes(page_bgr: np.ndarray, left_ratio=0.6):
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY) if page_bgr.ndim==3 else page_bgr
    bw = _binarize(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,3))
    merge = cv2.dilate(bw, kernel, 1)
    contours,_ = cv2.findContours(merge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H,W = bw.shape; max_x = int(W*left_ratio); boxes=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if x>max_x or h<16 or w<20 or w*h<200 or h>H*0.35: continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: (b[1]//40, b[0]))
    return boxes

# Детекция за задраскване: една дълга, почти хоризонтална линия ←→, поне 3 сегмента и в двете посоки
def is_crossed_out(word_img: np.ndarray) -> bool:
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY) if word_img.ndim==3 else word_img
    h, w = gray.shape[:2]
    if w < 30 or h < 15: return False
    edges = cv2.Canny(gray, 50, 150)
    min_len = max(12, int(0.65 * w))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=min_len, maxLineGap=12)
    if lines is None: return False
    horiz=[]
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        dx, dy = x2-x1, y2-y1
        length = max(1, int((dx*dx + dy*dy) ** 0.5))
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        if ang <= 15 or abs(ang-180) <= 15:
            horiz.append((dx,dy,length,y1,y2))
    if len(horiz) < 3: return False
    has_pos = any(dx > 0 for dx,_,_,_,_ in horiz)
    has_neg = any(dx < 0 for dx,_,_,_,_ in horiz)
    if not (has_pos and has_neg): return False
    ys = [y1 for _,_,_,y1,_ in horiz] + [y2 for _,_,_,_,y2 in horiz]
    band = max(ys)-min(ys) if ys else h
    if band > 0.45*h: return False
    med_len = float(np.median([L for *_,L,__,__ in [(dx,dy,L,y1,y2) for dx,dy,L,y1,y2 in horiz]]))
    if med_len < 0.70*w: return False
    return True

def mask_crossed_out_names(page_bgr: np.ndarray, left_ratio: float = 0.6):
    boxes = extract_word_boxes(page_bgr, left_ratio=left_ratio)
    out = page_bgr.copy()
    removed = []
    for (x,y,w,h) in boxes:
        crop = page_bgr[y:y+h, x:x+w]
        if is_crossed_out(crop):
            pad = 3
            cv2.rectangle(out, (max(0,x-pad), max(0,y-pad)), (min(out.shape[1]-1, x+w+pad), min(out.shape[0]-1, y+h+pad)), (255,255,255), thickness=-1)
            removed.append((x,y,w,h))
    return out, removed
""")

w("infra/ai/handwrite_ai.py", """from typing import List
import numpy as np
from core.ports import IHandwriteAI
# Засега не ползваме ML за имена (OCR ги хваща), оставено за бъдещо разширение.
class ProtoHandwriteAI(IHandwriteAI):
    def recognize_names(self, page_bgr: 'np.ndarray', known_names: List[str], min_conf: float = 0.6) -> List[str]:
        return []
""")

# ---------- UI ----------
w("ui_kivy/__init__.py", "")
w("ui_kivy/app.py", """from __future__ import annotations
from datetime import datetime
import os, time
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition
from kivy.graphics import Color, Rectangle, Line, InstructionGroup
from kivy.metrics import dp
from core.services import SavePageService

def _cm_to_dp(cm: float) -> float:
    from kivy.metrics import dp
    return dp(cm * 160.0 / 2.54)

def add_ruled_paper(widget, step_cm=0.7, line_alpha=0.07):
    # лек „тетрадка“ фон (бледи хоризонтални линии)
    group = InstructionGroup()
    with widget.canvas.before:
        Color(1,1,1,1); bg = Rectangle(pos=widget.pos, size=widget.size)
    widget._ruled_bg_rect = bg
    def redraw(*_):
        try: widget.canvas.before.remove(group)
        except Exception: pass
        group.clear()
        with widget.canvas.before:
            Color(1,1,1,1)
            widget._ruled_bg_rect.pos = widget.pos
            widget._ruled_bg_rect.size = widget.size
            y = widget.y; step = _cm_to_dp(step_cm)
            while y < widget.top:
                Color(0,0,0,line_alpha); group.add(Rectangle(pos=(widget.x, y), size=(widget.width, dp(1))))
                y += step
    widget.bind(pos=redraw, size=redraw); widget.canvas.before.add(group); redraw()

class DrawArea(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'; self.padding = dp(8); self.spacing = dp(8)
        self.canvas_box = BoxLayout(size_hint=(1,1)); self.add_widget(self.canvas_box)
        with self.canvas_box.canvas:
            Color(1,1,1,1); self.bg_rect = Rectangle(size=self.canvas_box.size, pos=self.canvas_box.pos)
        self.canvas_box.bind(size=self._update_bg, pos=self._update_bg)
        add_ruled_paper(self.canvas_box)
        self._lines = []
        self.canvas_box.bind(on_touch_down=self._down, on_touch_move=self._move)

    def _update_bg(self, *_):
        self.bg_rect.size = self.canvas_box.size; self.bg_rect.pos = self.canvas_box.pos

    def _down(self, w, t):
        if not self.canvas_box.collide_point(*t.pos): return
        with self.canvas_box.canvas: Color(0,0,0,1); self._lines.append(Line(points=[*t.pos], width=2))

    def _move(self, w, t):
        if not self.canvas_box.collide_point(*t.pos): return
        if self._lines: self._lines[-1].points += [*t.pos]

    def export_to_png(self, path: str):
        img = self.canvas_box.export_as_image(); img.save(path)

class SearchView(Screen):
    def __init__(self, repo, **kwargs):
        super().__init__(**kwargs); self.repo = repo
        root = BoxLayout(orientation='vertical')
        top = GridLayout(cols=3, size_hint=(1,None), height=dp(64), padding=[dp(12),0,dp(12),0], spacing=dp(8))
        with top.canvas.before: Color(0,0,0,1); self._r = Rectangle(size=top.size, pos=top.pos)
        top.bind(size=lambda *_: self._u(top), pos=lambda *_: self._u(top))
        btn_back = Button(text='← Платно', size_hint=(None,1), width=dp(160), background_normal='', background_color=(1,1,1,1), color=(0,0,0,1), font_size=dp(18))
        btn_back.bind(on_press=lambda *_: setattr(self.manager, 'current', 'draw'))
        top.add_widget(btn_back); top.add_widget(Label(text='Търсене', color=(1,1,1,1), font_size=dp(26))); top.add_widget(Label()); root.add_widget(top)
        bar = BoxLayout(orientation='horizontal', size_hint=(1,None), height=dp(64), padding=[dp(12), dp(8)], spacing=dp(8))
        self.search_input = TextInput(hint_text='🔍 Въведи име…', multiline=False, font_size=dp(18), background_normal='', background_color=(0.95,0.95,0.95,1), foreground_color=(0,0,0,1), padding=[dp(12)])
        btn = Button(text='Търси', size_hint=(None,1), width=dp(100), background_normal='', background_color=(0.2,0.6,1,1), color=(1,1,1,1), font_size=dp(18))
        btn.bind(on_press=self.on_search)
        bar.add_widget(self.search_input); bar.add_widget(btn); root.add_widget(bar)
        self.results = Label(text='', halign='left', valign='top', markup=True)
        self.results.bind(size=lambda *_: setattr(self.results,'text_size', self.results.size))
        sv = ScrollView(size_hint=(1,1)); sv.add_widget(self.results); root.add_widget(sv); self.add_widget(root)

    def _u(self, w): self._r.size = w.size; self._r.pos = w.pos

    def on_search(self, *_):
        q = (self.search_input.text or '').strip()
        rows = self.repo.search_by_name(q)
        if not rows:
            self.results.text = 'Нищо не е намерено.'; return
        lines = [f'• [b]{name}[/b] — {total/100.0:.2f} лв' for name, total in rows]
        self.results.text = '\\n'.join(lines)

class TrainerScreen(Screen):
    # оставен за бъдещо обучение; не се пуска автоматично
    pass

class DrawView(Screen):
    def __init__(self, service: SavePageService, **kwargs):
        super().__init__(**kwargs); self.service = service
        root = BoxLayout(orientation='vertical')
        top = GridLayout(cols=3, size_hint=(1,None), height=dp(64), padding=[dp(12),0,dp(12),0], spacing=dp(8))
        with top.canvas.before: Color(0,0.4,1,1); self._r = Rectangle(size=top.size, pos=top.pos)
        top.bind(size=lambda *_: self._u(top), pos=lambda *_: self._u(top))
        btn_save = Button(text='Запазване', size_hint=(None,1), width=dp(140), background_normal='', background_color=(1,0.5,0,1), color=(1,1,1,1), font_size=dp(18))
        btn_save.bind(on_press=self.on_save)
        top.add_widget(btn_save)
        top.add_widget(Label(text='Платно', color=(1,1,1,1), font_size=dp(26)))
        btn_search = Button(text='Търсене', size_hint=(None,1), width=dp(140), background_normal='', background_color=(0,0,0,0), color=(1,1,1,1), font_size=dp(18))
        btn_search.bind(on_press=lambda *_: setattr(self.manager, 'current', 'search'))
        top.add_widget(btn_search)
        root.add_widget(top)
        self.draw_area = DrawArea(); root.add_widget(self.draw_area)
        self.status = Label(text='Готово', size_hint=(1,None), height=dp(28), color=(0,0,0,1), font_size=dp(14), halign='left', valign='middle')
        root.add_widget(self.status)
        self.add_widget(root)

    def _u(self, w): self._r.size = w.size; self._r.pos = w.pos
    def set_status(self, msg): self.status.text = msg

    def on_save(self, *_):
        import time
        os.makedirs('pages', exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S'); path = os.path.join('pages', f'page_{ts}.png')
        try:
            self.draw_area.export_to_png(path)
        except Exception as e:
            self.set_status(f'Грешка при запазване на PNG: {e}'); return

        from datetime import datetime
        page_id, n = self.service.save_drawn_page(path, datetime.now().isoformat(timespec='seconds'))

        # изчистване на платното
        self.draw_area.canvas_box.canvas.clear()
        with self.draw_area.canvas_box.canvas:
            Color(1,1,1,1); self.draw_area.bg_rect = Rectangle(size=self.draw_area.canvas_box.size, pos=self.draw_area.canvas_box.pos)
        add_ruled_paper(self.draw_area.canvas_box)

        skipped = getattr(self.service, 'last_removed_count', 0)
        import os as _os
        self.set_status(f'Запазени {n} реда. Пропуснати (задраскани): {skipped}. PNG: {_os.path.basename(path)}')

class VeresiyaApp(App):
    def __init__(self, repo, service: SavePageService, **kwargs):
        super().__init__(**kwargs); self.repo = repo; self.service = service
    def build(self):
        from kivy.uix.screenmanager import ScreenManager, FadeTransition
        sm = ScreenManager(transition=FadeTransition())
        self.repo.init()
        sm.add_widget(DrawView(self.service, name='draw'))
        from infra.database_sqlite import SQLiteRepo
        sm.add_widget(SearchView(self.repo, name='search'))
        sm.current = 'draw'
        return sm
""")

# ---------- entry ----------
w("main.py", """from infra.database_sqlite import SQLiteRepo
from infra.ocr_bul import TesseractOCR
from infra.ai.handwrite_ai import ProtoHandwriteAI
from core.services import SavePageService
from ui_kivy.app import VeresiyaApp

repo = SQLiteRepo()
ocr = TesseractOCR()
ai = ProtoHandwriteAI()
service = SavePageService(repo=repo, ai=ai, ocr=ocr)

if __name__ == '__main__':
    VeresiyaApp(repo=repo, service=service).run()
""")

# ---------- buildozer.spec ----------
w("buildozer.spec", """[app]
title = Veresia
package.name = veresia
package.domain = com.yourstore
source.dir = .
source.include_exts = py,kv,png,jpg,ttf,zip,txt,md,npz,db
version = 0.1
requirements = python3,kivy,numpy,opencv,pillow,plyer,pytesseract
orientation = sensor
fullscreen = 0
android.archs = arm64-v8a, armeabi-v7a
android.api = 33
android.minapi = 21
p4a.branch = master
android.permissions =
""")

print("✅ Project generated.")
