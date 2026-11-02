"""Record the silent public Pramaan demo in a fresh anonymous Playwright context."""
from __future__ import annotations
import argparse, json, re, urllib.request
from pathlib import Path
from urllib.parse import urlsplit

PUBLIC_ORIGIN = "https://kshitijthakkar-pramaan-proof-of-art.hf.space"
LOOPBACK_HOST = ".".join(("127", "0", "0", "1"))
TOKEN_PATTERNS = [re.compile(r"(?i)\bhf_[a-z0-9]{20,}\b"), re.compile(r"(?i)\bsk-ant-[a-z0-9_-]{20,}\b"), re.compile(r"(?i)\bsk-[a-z0-9_-]{24,}\b"), re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|authorization)\s*[:=]\s*(?:bearer\s+)?[a-z0-9._-]{16,}")]
RANGOLI_TERMS = ("rangoli", "kolam", "pookalam", "aipan", "alpana", "muggu", "chowk", "sathiya")

def validate_public_url(url: str, expected_origin: str = PUBLIC_ORIGIN) -> str:
    p, expected = urlsplit(url), urlsplit(expected_origin)
    if p.scheme != "https" or p.username or p.password or p.port or p.netloc != expected.netloc or p.hostname != expected.hostname:
        raise ValueError(f"Refusing URL outside public Space origin: {p.scheme}://{p.netloc}")
    return url

def _recording_origin(url: str, *, allow_local: bool = False) -> str:
    """Return the exact permitted origin; local recording is loopback-only and opt-in."""
    parsed = urlsplit(url.rstrip("/"))
    if parsed.scheme == "https" and parsed.netloc == urlsplit(PUBLIC_ORIGIN).netloc:
        if parsed.username or parsed.password or parsed.port:
            raise ValueError("Public recording URL must use the canonical HTTPS origin")
        return f"{parsed.scheme}://{parsed.netloc}"
    if (allow_local and parsed.scheme == "http" and parsed.hostname in {"localhost", LOOPBACK_HOST}
            and parsed.port and not parsed.username and not parsed.password):
        return f"{parsed.scheme}://{parsed.netloc}"
    raise ValueError("Recording URL must be the public Space, or an explicitly allowed loopback URL")

def _validate_recording_url(url: str, expected_origin: str) -> str:
    parsed, expected = urlsplit(url), urlsplit(expected_origin)
    if parsed.scheme != expected.scheme or parsed.netloc != expected.netloc or parsed.username or parsed.password:
        raise ValueError(f"Refusing URL outside allowed recording origin: {parsed.scheme}://{parsed.netloc}")
    if expected_origin == f"https://{urlsplit(PUBLIC_ORIGIN).netloc}":
        return validate_public_url(url)
    if expected.scheme != "http" or expected.hostname not in {"localhost", LOOPBACK_HOST}:
        raise ValueError("Non-public recording origin must be loopback")
    return url

def contains_token_like_string(text: str) -> bool:
    return any(p.search(text or "") for p in TOKEN_PATTERNS)

def is_non_rangoli_state(state: str, tradition: str = "") -> bool:
    label = f"{state} {tradition}".casefold()
    return not any(term in label for term in RANGOLI_TERMS)

def build_live_caption(template: str, summary: dict[str, object]) -> str:
    if contains_token_like_string(template): raise ValueError("Caption contains token-like text")
    try: caption = template.format_map(summary)
    except KeyError as exc: raise ValueError(f"Caption value missing from live summary: {exc.args[0]}") from exc
    if contains_token_like_string(caption): raise ValueError("Caption contains token-like text")
    return caption

def warm_space(base_url: str = PUBLIC_ORIGIN) -> None:
    for _ in range(2):
        with urllib.request.urlopen(urllib.request.Request(base_url.rstrip("/") + "/health", headers={"User-Agent":"PramaanDemoRecorder/1.0"}), timeout=20) as r:
            if not 200 <= r.status < 300: raise RuntimeError(f"Space health check failed: HTTP {r.status}")

_OVERLAY_JS = """(()=>{
const s=document.getElementById('poa-demo-style')||document.createElement('style');
s.id='poa-demo-style';
s.textContent=`#poa-demo-cursor{position:fixed;z-index:2147483647;width:16px;height:16px;border-radius:50%;background:#ffd166;border:2px solid #172b4d;box-shadow:0 1px 8px #1118;pointer-events:none;transform:translate(-50%,-50%)}.poa-demo-ripple{position:fixed;z-index:2147483646;width:28px;height:28px;border:3px solid #ffd166;border-radius:50%;pointer-events:none;transform:translate(-50%,-50%);animation:poa-ripple .55s ease-out forwards}@keyframes poa-ripple{to{opacity:0;scale:2.1}}#poa-demo-caption{position:fixed;z-index:2147483645;left:5%;right:5%;bottom:4%;padding:14px 22px;border-radius:12px;background:#101b2de8;color:white;font:700 clamp(22px,3vw,38px)/1.2 system-ui,sans-serif;text-align:center;pointer-events:none}`;
(document.head||document.documentElement).appendChild(s);
const add=()=>{if(!document.body)return;if(!document.getElementById('poa-demo-cursor')){const c=document.createElement('i');c.id='poa-demo-cursor';document.body.appendChild(c)}if(!document.getElementById('poa-demo-caption')){const c=document.createElement('div');c.id='poa-demo-caption';document.body.appendChild(c)}};
window.__poaCaption=t=>{add();const c=document.getElementById('poa-demo-caption');if(c)c.textContent=t||''};
window.__poaCursor=(x,y)=>{add();const c=document.getElementById('poa-demo-cursor');if(c){c.style.left=x+'px';c.style.top=y+'px'}};
window.__poaRipple=(x,y)=>{add();const e=document.createElement('i');e.className='poa-demo-ripple';e.style.left=x+'px';e.style.top=y+'px';document.body.appendChild(e);setTimeout(()=>e.remove(),700)};
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',add,{once:true});else add()
})()"""

def _ensure_overlay(page):
    page.evaluate(_OVERLAY_JS)

def _overlay(page):
    page.add_init_script(_OVERLAY_JS)
    _ensure_overlay(page)

def _caption(page, value):
    if contains_token_like_string(value): raise RuntimeError("Token-like caption")
    _ensure_overlay(page)
    page.evaluate("t=>window.__poaCaption(t)", value)

def _wait_for_loaded_surface(page, selector: str, timeout: int = 45000) -> None:
    """Wait for the interactive surface; Gradio keeps network activity open."""
    page.locator(selector).wait_for(state="visible", timeout=timeout)

def _check_page(page, origins, expected_origin=PUBLIC_ORIGIN):
    u = _validate_recording_url(page.url, expected_origin); p = urlsplit(u); origins.append(p.scheme + "://" + p.netloc)
    if _page_requires_sign_in(page): raise RuntimeError("Page requires sign-in")
    if contains_token_like_string(page.locator("body").inner_text(timeout=5000)): raise RuntimeError("Token-like string in DOM")

def _page_requires_sign_in(page) -> bool:
    """Detect a login gate, not ordinary copy such as 'Sign in to publish'."""
    return bool(page.locator('input[type="password"]:visible').count()
                or page.locator('form[action*="/login"]:visible,form[action*="oauth"]:visible').count())

def _move_click(page, loc):
    loc.wait_for(state="visible", timeout=15000); b=loc.bounding_box()
    if not b: raise RuntimeError("Target has no visible bounds")
    _ensure_overlay(page)
    x,y=b["x"]+b["width"]/2,b["y"]+b["height"]/2
    for i in range(1,13):
        t=i/12; e=t*t*(3-2*t); sx,sy=640,500
        page.mouse.move(sx+(x-sx)*e,sy+(y-sy)*e); page.evaluate('({x,y})=>window.__poaCursor(x,y)',{'x':sx+(x-sx)*e,'y':sy+(y-sy)*e})
    page.evaluate("({x,y})=>{window.__poaCursor(x,y);window.__poaRipple(x,y)}", {"x":x,"y":y})
    loc.click()

def _hold(page, seconds, frame_dir, name):
    from io import BytesIO
    from PIL import Image
    import math
    for i in range(math.ceil(seconds/2)):
        _check_page(page, [])
        page.wait_for_timeout(2000)
        Image.open(BytesIO(page.screenshot(type="png"))).save(frame_dir/f"{name}-{i:02}.png")

def _tab(page, label):
    tab=page.locator("a[data-tab-button]").filter(has_text=re.compile(label,re.I)).first
    _move_click(page,tab)
    page.wait_for_function("l=>[...document.querySelectorAll('a[data-tab-button]')].some(x=>x.textContent.match(new RegExp(l,'i'))&&x.getAttribute('aria-selected')==='true')",arg=label,timeout=15000)

def _summary(page):
    rows=page.locator('[data-leaderboard-row][data-source="official"]')
    models=rows.evaluate_all("es=>es.map(e=>e.dataset.modelKey).filter(Boolean)")
    if not models: raise RuntimeError("No live official leaderboard summary available")
    return {"model_count":len(models),"models":models}

SAFE_WALL_SCENES = (
    ("Gujarat", "gj_bandhani-easy-6", "home-wall-tile-gujarat"),
    ("Nagaland", "nl_chakhesang-shawls-easy-17", "home-wall-tile-nagaland"),
    ("Delhi", "dl_mughal-sandstone-and-marble-jaali-screens-of-the-humayun-s-tomb-complex-easy-31",
     "home-wall-tile-delhi"),
)


def _safe_wall_scenes(page):
    """Return only the three human-reviewed Home Wall tiles for the demo."""
    tiles = page.locator('#state-wall [data-home-wall-tile]')
    if tiles.count() != 36:
        raise RuntimeError(f"Home Wall must show 36 state tiles; found {tiles.count()}")
    scenes = []
    for state, task_id, test_id in SAFE_WALL_SCENES:
        tile = page.get_by_test_id(test_id)
        if tile.count() != 1:
            raise RuntimeError(f"Safe demo tile is missing or duplicated: {state}")
        actual = tile.get_attribute('data-easy-task-id')
        if actual != task_id:
            raise RuntimeError(f"Safe demo task changed for {state}: expected {task_id}, found {actual}")
        scenes.append((state, task_id, tile))
    return scenes

def _contact_sheet(frame_dir, target):
    from PIL import Image,ImageDraw,ImageOps
    paths=sorted(frame_dir.glob('*.png'))
    if not paths: raise RuntimeError("No two-second frame samples")
    cols=3; sheet=Image.new('RGB',(960,((len(paths)+2)//3)*205),'#e9edf3'); draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(paths):
        with Image.open(p) as im: thumb=ImageOps.pad(im.convert('RGB'),(320,180))
        x,y=(i%cols)*320,(i//cols)*205; sheet.paste(thumb,(x,y)); draw.text((x+8,y+184),p.stem,fill='#172b4d')
    target.parent.mkdir(parents=True,exist_ok=True); sheet.save(target)


def _make_derivatives(frame_dir, output_dir):
    """Generate a clean poster and silent compact GIF from the captured two-second frames."""
    from PIL import Image
    files=sorted(frame_dir.glob('*.png'))
    if not files: raise RuntimeError('No frames available for poster/GIF')
    root=Path(output_dir).parent
    poster=root/'media'/'pramaan-demo-poster.png'
    poster.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(files[0]) as im: im.convert('RGB').save(poster,format='PNG',optimize=True)
    frames=[]
    for path in files:
        with Image.open(path) as im:
            frame=im.convert('RGB'); frame.thumbnail((640,360)); frame=frame.quantize(colors=128,method=Image.Quantize.MEDIANCUT)
            frames.append(frame)
    gif=root/'assets'/'demo.gif'; gif.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(gif,format='GIF',save_all=True,append_images=frames[1:],duration=2000,loop=0,optimize=True,disposal=2)
    return poster,gif

def _maybe_encode_h264(webm):
    import shutil, subprocess
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg: return webm
    mp4=webm.with_name('pramaan-demo.mp4')
    result=subprocess.run([ffmpeg,'-y','-i',str(webm),'-map_metadata','-1','-map_chapters','-1','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart','-r','30','-crf','23',str(mp4)],capture_output=True,text=True)
    if result.returncode: raise RuntimeError('H.264 encoding failed: '+result.stderr[-1000:])
    return mp4
def record(output_dir: Path, base_url: str = PUBLIC_ORIGIN, *, allow_local: bool = False) -> Path:
    """Record a silent draft WebM from a fresh anonymous context; never signs in or uses tokens."""
    expected_origin = _recording_origin(base_url, allow_local=allow_local)
    base_url=_validate_recording_url(base_url.rstrip('/'), expected_origin); warm_space(base_url)
    try: from playwright.sync_api import sync_playwright
    except ImportError as exc: raise RuntimeError("Playwright is required to record") from exc
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    frame_dir=output_dir.parent/'local'/'demo-frames'; frame_dir.mkdir(parents=True,exist_ok=True)
    origins=[]; violations=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1280,'height':720},device_scale_factor=1,record_video_dir=str(output_dir),record_video_size={'width':1280,'height':720})
        page=context.new_page(); _overlay(page)
        def nav(frame):
            if frame==page.main_frame:
                try: _validate_recording_url(frame.url, expected_origin)
                except Exception as e: violations.append(e); raise
        page.on('framenavigated',nav)
        page.on('console',lambda msg: violations.append(RuntimeError('Token-like console content')) if contains_token_like_string(msg.text) else None)
        page.goto(_validate_recording_url(base_url+'/studio/',expected_origin),wait_until='domcontentloaded',timeout=45000); _wait_for_loaded_surface(page,'[data-testid="proof-replay"]'); _check_page(page,origins,expected_origin)
        _caption(page,'AI drew it. Math checked it. Proof of art that you can verify.')
        proof=page.locator('[data-testid="proof-replay"]').first
        if not proof.count(): raise RuntimeError('Proof Replay hero is unavailable on the deployed Space')
        proof.wait_for(state='visible',timeout=15000)
        step=proof.get_by_role('button',name=re.compile('next|step|play',re.I)).first
        if not step.count(): raise RuntimeError('Proof Replay has no visible step control')
        _move_click(page,step)
        page.wait_for_function("document.querySelector('[data-testid=proof-replay]')?.innerText",timeout=10000)
        _hold(page,8,frame_dir,'proof')
        _tab(page,'Home')
        scenes=_safe_wall_scenes(page)
        wall=page.locator('#state-wall')
        if wall.locator('[data-wall-mode-choice="attempt"]').count():
            _move_click(page,wall.locator('[data-wall-mode-choice="attempt"]'))
            page.wait_for_function("document.querySelector('#state-wall')?.dataset.wallDisplay==='attempt'",timeout=5000)
        _caption(page,build_live_caption('{state_count} regional views. One living design language.',{'state_count':page.locator('#state-wall [data-home-wall-tile]').count()})); _hold(page,4,frame_dir,'wall-map')
        for state, task_id, tile in scenes:
            tile.hover(); _move_click(page,tile)
            dialog=page.locator('[data-wall-dialog]')
            dialog.wait_for(state='visible',timeout=10000)
            easy=dialog.locator('[data-wall-task-level="easy"]')
            details=easy.inner_text()
            if state=='Gujarat' and not re.search(r'Attempt\s+3:\s+passed',details,re.I):
                raise RuntimeError('Gujarat safe scene no longer shows the accepted repair')
            if state=='Nagaland' and not re.search(r'Attempt\s+2:\s+passed',details,re.I):
                raise RuntimeError('Nagaland safe scene no longer shows accepted attempt 2')
            attempt_image=easy.locator('img').nth(1)
            page.wait_for_function('img=>img.complete&&img.naturalWidth>0',arg=attempt_image.element_handle(),timeout=10000)
            caption={
                'Gujarat':'Gujarat · Bandhani, verified after repair',
                'Nagaland':'Nagaland · accepted attempt 2',
                'Delhi':'Delhi · Mughal jaali screen',
            }[state]
            _caption(page,caption); _hold(page,4,frame_dir,f'wall-{state.casefold()}')
            _move_click(page,dialog.get_by_test_id('home-wall-state-sheet-close'))
        _tab(page,'Models'); page.wait_for_selector('[data-leaderboard-row][data-source="official"]',timeout=15000)
        summary=_summary(page); _caption(page,build_live_caption('{model_count} models compared on one shared suite',summary)); _hold(page,15,frame_dir,'arena')
        page.goto(_validate_recording_url(base_url+'/web/',expected_origin),wait_until='domcontentloaded',timeout=45000); _wait_for_loaded_surface(page,'#poa-pattern'); _check_page(page,origins,expected_origin)
        reset=page.locator('#poa-reset'); _move_click(page,reset); page.wait_for_function("document.querySelector('#poa-status')?.innerText",timeout=15000)
        passing=page.locator('#poa-pass')
        if passing.count(): _move_click(page,passing); page.wait_for_function("document.querySelector('#poa-pattern')?.value",timeout=15000)
        _move_click(page,page.locator('#poa-run')); page.wait_for_function("/reward\\s*[:=]?\\s*1(?:\\.0)?/i.test(document.querySelector('#poa-status')?.innerText||'')",timeout=20000)
        text=page.locator('#poa-status').inner_text(); m=re.search(r'reward\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)',text,re.I)
        if not m: raise RuntimeError('Could not read live OpenEnv reward')
        _caption(page,build_live_caption('OpenEnv verifier reward: {reward}',{'reward':m.group(1)})); _hold(page,12,frame_dir,'openenv')
        page.goto(_validate_recording_url(base_url+'/studio/',expected_origin),wait_until='domcontentloaded',timeout=45000); _wait_for_loaded_surface(page,'[data-testid="proof-replay"]'); _check_page(page,origins,expected_origin)
        _caption(page,'Explore the Space, collection and dataset.'); _hold(page,9,frame_dir,'close')
        _caption(page,'Happy Diwali'); _hold(page,2,frame_dir,'diwali')
        if violations: raise RuntimeError('Recording aborted: '+str(violations[0]))
        video=page.video; context.close()
        if video is None: raise RuntimeError('Playwright produced no video')
        target=output_dir/'pramaan-demo.webm'; video.save_as(target)
        browser.close()
    # Store origins only, never full URLs, queries, cookies, or page state.
    (output_dir.parent/'local'/'demo-url-origins.json').write_text(json.dumps(sorted(set(origins)),indent=2)+'\n',encoding='utf-8')
    _contact_sheet(frame_dir,output_dir.parent/'local'/'demo-contact-sheet.png')
    return _maybe_encode_h264(target)

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--url',default=PUBLIC_ORIGIN); parser.add_argument('--output-dir',type=Path,default=Path('media')); parser.add_argument('--allow-local',action='store_true',help='allow only localhost for a private dry run'); a=parser.parse_args()
    try: print('Draft recording:',record(a.output_dir,a.url,allow_local=a.allow_local))
    except Exception as e: parser.exit(1,f'Recording failed safely: {e}\n')
    return 0
if __name__=='__main__': raise SystemExit(main())


