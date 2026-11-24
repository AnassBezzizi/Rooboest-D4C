"""Council of Achrona: een mini-Council met Raadsleden en een Koning.

Start met `uv run council`. Schrijft verdict.json en trace.json in de root van de repo.
De Koning leidt het debat in `kies_volgende()`: hij beslist wie spreekt, met welke vraag en waarom.
"""
import asyncio
import json
import os
import sys
import uuid
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

from agent_framework import Agent, MCPStreamableHTTPTool
from agent_framework.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

import raadsleden

ROOT = Path(__file__).parent
KENNISBRONNEN = ROOT / "kennisbronnen"
ROLLEN = ("archmage", "engineer", "guardian_keeper", "historian", "oracle", "beast_master")
BEURTENBUDGET = 10  # spelregel: maximaal 10 Raadslid-beurten (berichten van Raadsleden) per run


def _vaste_volgorde(beurten, leden, extra_reden=""):
    """Fallback: wie is er in alfabetische volgorde aan de beurt? Gebruikt als de LLM-Koning faalt."""
    rondes = int(os.getenv("RONDES", "2"))
    gesproken = sum(1 for b in beurten if b["soort"] == "bericht" and b["spreker"] in ROLLEN)
    if gesproken >= rondes * len(leden):
        return None, None, f"vaste volgorde: alle rondes gespeeld{(' — ' + extra_reden) if extra_reden else ''}"
    lid = leden[gesproken % len(leden)]
    return lid.ROL, "Geef je inbreng in hoogstens 150 woorden. Reageer op wat de anderen zeiden.", f"vaste volgorde{(' (' + extra_reden + ')') if extra_reden else ''}"


def kies_volgende(beurten, leden):
    """De Koning leidt het debat: hij beslist wie spreekt. De kit roept deze functie op vóór elke beurt.
    (In AI-termen heet dit deel een orchestrator.)

    De Koning is hier een LLM-agent: hij leest het debat tot nu toe en kiest bewust wie spreekt,
    met welke vraag en waarom. Hij kan ook zelf stoppen als de Council genoeg weet voor een Verdict,
    vóór het Beurtenbudget op is. Bij een netwerkfout of ongeldig antwoord valt hij terug op de
    alfabetische vaste volgorde.
    """
    gesproken_berichten = [b for b in beurten if b["soort"] == "bericht" and b["spreker"] in ROLLEN]
    gesproken = len(gesproken_berichten)
    gesproken_rollen = {b["spreker"] for b in gesproken_berichten}
    aanwezige_rollen = [l.ROL for l in leden]
    niet_gesproken = [r for r in aanwezige_rollen if r not in gesproken_rollen]
    beurten_over = BEURTENBUDGET - gesproken

    # Eerste beurt: altijd het eerste lid (alfabetisch), om context op te bouwen.
    if gesproken == 0:
        lid = leden[0]
        return (
            lid.ROL,
            "Introduceer je perspectief en vertel wat jouw Kennisbron over de Desync zegt. "
            "Noem minstens twee concrete record-ID's uit je bron.",
            "eerste beurt: context opbouwen vanaf de eerste Rol",
        )

    # Bouw een beknopt overzicht van het debat voor de Koning.
    debat_regels = []
    for b in gesproken_berichten:
        tekst = b["tekst"]
        if len(tekst) > 700:
            tekst = tekst[:700] + "..."
        debat_regels.append(f"[{b['spreker']}]: {tekst}")

    rollen_tekst = "\n".join(f"- {l.ROL} ({l.NAAM}): {l.PERSPECTIEF}" for l in leden)
    niet_gesproken_tekst = ", ".join(niet_gesproken) if niet_gesproken else "(iedereen heeft al minstens één keer gesproken)"

    prompt = (
        "Je bent de Koning van de Council of Achrona. Je leidt het debat over de Desync, maar spreekt zelf niet "
        "en leest geen Kennisbron. Je taak: kies voor de volgende beurt het Raadslid dat de Council het meest verder helpt, "
        "of stop het debat als de Council genoeg weet voor een Verdict.\n\n"
        f"Beschikbare Raadsleden:\n{rollen_tekst}\n\n"
        f"Raadsleden die nog NIET gesproken hebben: {niet_gesproken_tekst}\n"
        f"Resterend Beurtenbudget: {beurten_over} Raadslid-beurten.\n\n"
        "Debat tot nu toe:\n\n" + "\n\n".join(debat_regels) + "\n\n"
        "Kies bewust. Weeg mee:\n"
        "- Welke beweringen zijn nog niet door een ander Raadslid getoetst?\n"
        "- Naar welke Rol verwees iemand, of wiens kennis ontbreekt nog?\n"
        "- Welke delen van de oorzaak (ingreep, mechanisme, verloop) zijn nog onbekend?\n"
        "- Herhaalt een Raadslid alleen wat er al staat? Kies dan iemand anders.\n"
        "- Als de Council de oorzaak, het herstelplan en de risico's helder genoeg heeft, stop dan (rol = null).\n\n"
        "Antwoord ALLEEN met geldige JSON in exact dit formaat (geen tekst eromheen, geen markdown-codefences):\n"
        '{"rol": "<rolnaam uit de lijst hierboven, of null om te stoppen>", '
        '"vraag": "<de concrete vraag voor dat Raadslid, maximaal 2 zinnen>", '
        '"reden": "<waarom je deze keuze maakt, maximaal 1 zin>"}'
    )

    try:
        sync_client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=3,
        )
        r = sync_client.chat.completions.create(
            model=os.environ["MODEL_GROOT"],
            messages=[{"role": "user", "content": prompt}],
        )
        ruw = (r.choices[0].message.content or "").strip()
    except Exception as e:  # netwerkfout, rate limit uitgeput, etc.
        return _vaste_volgorde(beurten, leden, extra_reden=f"LLM-fout: {e.__class__.__name__}")

    # Modellen zetten soms tekst of markdown rond hun JSON: knip naar { ... }.
    start, eind = ruw.find("{"), ruw.rfind("}")
    if start == -1 or eind == -1:
        return _vaste_volgorde(beurten, leden, extra_reden="Koning gaf geen JSON terug")
    try:
        keuze = json.loads(ruw[start:eind + 1])
    except json.JSONDecodeError:
        return _vaste_volgorde(beurten, leden, extra_reden="Koning gaf ongeldige JSON")

    rol = keuze.get("rol")
    vraag = (keuze.get("vraag") or "").strip() or "Geef je inbreng in hoogstens 150 woorden en reageer op wat er al gezegd is."
    reden = (keuze.get("reden") or "").strip() or "Koning kiest"

    # Stoppen: rol = null, "null", lege string of iets dat niet in de lijst zit.
    if rol in (None, "", "null"):
        return None, None, reden

    if rol not in aanwezige_rollen:
        return _vaste_volgorde(beurten, leden, extra_reden=f"Koning koos onbekende rol {rol!r}")

    return rol, vraag, reden


def controleer(leden):
    """Duidelijke melding in plaats van een traceback bij de meest voorkomende fouten."""
    mist = [v for v in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_KLEIN", "MODEL_GROOT") if not os.getenv(v)]
    if mist:
        sys.exit(f"In .env ontbreekt: {', '.join(mist)}. Kopieer .env.example naar .env en vul het in, of zie 'Vooraf installeren' in de README.")
    for lid in leden:
        if lid.ROL not in ROLLEN:
            sys.exit(f"raadsleden/: ROL = {lid.ROL!r} is ongeldig. Gebruik de mapnaam van de Rol: {', '.join(ROLLEN)}.")
    dubbel = {l.ROL for l in leden if [x.ROL for x in leden].count(l.ROL) > 1}
    if dubbel:
        sys.exit(f"Eén Raadslid per Rol: {', '.join(dubbel)} komt meer dan eens voor in raadsleden/.")


def client(model_env):
    key = os.environ["OPENAI_API_KEY"]
    return OpenAIChatCompletionClient(
        model=os.environ[model_env],
        async_client=AsyncOpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=key,
            max_retries=8,  # bij een rate limit (429) wacht de client en probeert opnieuw
        ),
    )


def kennisbron_bestanden(rol):
    map_ = KENNISBRONNEN / rol
    return sorted(p for p in map_.glob("*") if p.suffix in (".md", ".csv")) if map_.is_dir() else []


def instructies(lid, bestanden, mcp):
    tekst = (
        f"Je bent de {lid.NAAM}, een Raadslid in de Council of Achrona. {lid.PERSPECTIEF}\n"
        "Je debatteert met de andere Raadsleden over de oorzaak van de Desync. "
        "Baseer je alleen op je eigen Kennisbron en noem bij elke bewering de record-ID's precies zoals ze in je Kennisbron staan. Verzin nooit een ID. "
        "Zeg het eerlijk als je iets niet weet."
    )
    if mcp:
        tekst += "\nLees of doorzoek je Kennisbron met je tools voor je spreekt."
    elif bestanden:
        tekst += "\n\nJe Kennisbron:\n\n" + "\n\n".join(f"## {p.name}\n\n{p.read_text(encoding='utf-8')}" for p in bestanden)
    else:
        tekst += "\nJe hebt nog geen Kennisbron: zeg dat je niets kan bewijzen."
    return tekst


def tel(verbruik, model, usage):
    if not usage:
        return
    v = verbruik.setdefault(model, {"beurten": 0, "invoer": 0, "gecached": 0, "uitvoer": 0})
    v["beurten"] += 1  # één beurt = één agent-run (met MCP soms meerdere model-requests)
    v["invoer"] += usage.get("input_token_count") or 0
    v["gecached"] += usage.get("cache_read_input_token_count") or 0
    v["uitvoer"] += usage.get("output_token_count") or 0


def toolcalls(response):
    for msg in response.messages:
        for c in msg.contents:
            if c.type == "function_call":
                args = c.arguments
                yield c.name, json.loads(args) if isinstance(args, str) and args else (args or {})


async def run():
    load_dotenv(ROOT / ".env")
    opties = {"reasoning_effort": os.environ["REASONING"]} if os.getenv("REASONING") else None  # minder denkwerk = minder tokens
    run_id = f"{datetime.now().astimezone().isoformat(timespec='seconds')}_{uuid.uuid4().hex[:8]}"
    leden = raadsleden.alle()
    controleer(leden)
    beurten, verbruik = [], {}
    trace = {"run_id": run_id, "status": "bezig", "raadsleden": [l.ROL for l in leden], "beurten": beurten, "geschillen": [], "verbruik": verbruik}

    def bewaar(naam, data):
        (ROOT / naam).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def beurt(ronde, spreker, soort, **velden):
        beurten.append({"nr": len(beurten) + 1, "ronde": ronde, "spreker": spreker, "soort": soort, **velden})
        bewaar("trace.json", trace)  # na elke beurt: de viewer (`uv run viewer`) toont het debat live

    klein = client("MODEL_KLEIN")

    async with AsyncExitStack() as stack:
        agents = []
        for lid in leden:
            mcp_url = os.getenv(f"MCP_{lid.ROL.upper()}", "").strip()
            bestanden = [] if mcp_url else kennisbron_bestanden(lid.ROL)
            tools = []
            if mcp_url:
                try:
                    tools.append(await stack.enter_async_context(MCPStreamableHTTPTool(name=f"kennisbron_{lid.ROL}", url=mcp_url)))
                except Exception as e:
                    sys.exit(f"MCP voor {lid.NAAM} ({mcp_url}) is onbereikbaar: {e}\n"
                             f"Is de server al open (vanaf 10:00)? Klopt het pad /{lid.ROL}/mcp? "
                             f"Of laat MCP_{lid.ROL.upper()} leeg om de bestanden uit kennisbronnen/{lid.ROL}/ te gebruiken.")
            elif not bestanden:
                print(f"LET OP: {lid.NAAM} heeft geen Kennisbron (MCP_{lid.ROL.upper()} is leeg en kennisbronnen/{lid.ROL}/ ook).")
            for p in bestanden:  # fallback zonder MCP: welke bestanden gingen in de prompt
                beurt(0, lid.ROL, "bestand", bestand=f"{lid.ROL}/{p.name}")
            agents.append((lid, mcp_url, Agent(klein, instructies(lid, bestanden, mcp_url), name=lid.ROL, tools=tools)))

        (ROOT / "verdict.json").unlink(missing_ok=True)  # setup gelukt: geen Verdict van een vorige run laten staan
        bewaar("trace.json", trace)
        per_rol = {lid.ROL: (lid, mcp_url, agent) for lid, mcp_url, agent in agents}
        debat, ronde = [], 1
        while True:
            gesproken = sum(1 for b in beurten if b["soort"] == "bericht" and b["spreker"] in per_rol)
            if gesproken >= BEURTENBUDGET:
                rol, vraag, reden = None, None, f"Beurtenbudget op ({BEURTENBUDGET} Raadslid-beurten)"
            else:
                rol, vraag, reden = kies_volgende(beurten, leden)
            if rol is not None and rol not in per_rol:
                sys.exit(f"kies_volgende() koos {rol!r}, maar die Rol zit niet in raadsleden/.")
            if rol is not None:
                ronde = gesproken // len(leden) + 1
            # Elke keuze van de Koning staat in de trace: aan wie, welke vraag en waarom.
            beurt(ronde, "koning", "routering", aan=rol, vraag=vraag, reden=reden)
            print(f"\n[Koning] {'stopt' if rol is None else '→ ' + per_rol[rol][0].NAAM}: {reden}")
            if rol is None:
                break
            lid, mcp_url, agent = per_rol[rol]
            # Wie ziet wat? Hier ziet elk Raadslid het hele debat (broadcast). Ook dat is een keuze.
            prompt = "Debat tot nu toe:\n\n" + ("\n\n".join(debat) or "(nog niets)") + f"\n\nVraag aan jou: {vraag}"
            trace["aan_het_woord"] = rol
            bewaar("trace.json", trace)
            r = await agent.run(prompt, options=opties)
            tel(verbruik, "klein", r.usage_details)
            for tool, args in toolcalls(r):
                beurt(ronde, rol, "toolcall", tool=tool, argumenten=args, pad=mcp_url)
            beurt(ronde, rol, "bericht", tekst=r.text)
            debat.append(f"{lid.NAAM}: {r.text}")
            print(f"\n[ronde {ronde}] {lid.NAAM}:\n{r.text}")

    # De Koning heeft geen Rol en geen Kennisbron: hij kent alleen wat de Raadsleden inbrachten.
    trace["aan_het_woord"] = "koning"
    bewaar("trace.json", trace)
    koning = Agent(client("MODEL_GROOT"), "Je bent de Koning van de Council of Achrona. Je hebt geen Kennisbron.", name="koning")
    r = await koning.run(
        options=opties,
        messages="Debat:\n\n" + "\n\n".join(debat)
        + "\n\nVat samen: waarover zijn de Raadsleden het eens, waarover niet, en welk bewijs (record-ID's) noemden ze?",
    )
    tel(verbruik, "groot", r.usage_details)
    beurt(ronde + 1, "koning", "bericht", tekst=r.text)
    print(f"\n[Koning]:\n{r.text}")

    # Stap 1: Geschillen extractie — waar spraken Raadsleden elkaar tegen, en hoe werd het beslecht?
    geschillen_prompt = (
        "Je bent de Koning van de Council of Achrona. Lees het debat hieronder en identificeer de GESCHILLEN: "
        "concrete punten waarop twee of meer Raadsleden het oneens waren, of een bewering die een ander tegensprak.\n\n"
        "Antwoord ALLEEN met geldige JSON, een lijst (eventueel leeg) in exact dit formaat (geen tekst eromheen, "
        "geen markdown-codefences):\n"
        "[\n"
        "  {\n"
        '    "onderwerp": "<waar ging het geschil over, één zin>",\n'
        '    "standpunten": [\n'
        '      {"rol": "<rolnaam>", "standpunt": "<korte samenvatting, max 1 zin>", "bewijs": ["<record-ID>", ...]}\n'
        "    ],\n"
        '    "beslechting": "<hoe werd het beslecht in het debat, of \'onopgelost\' als er geen consensus kwam>"\n'
        "  }\n"
        "]\n\n"
        "Regels:\n"
        "- Alleen echte tegenspraken. Gebruik geen geschil om dingen op te vullen die niemand tegensprak.\n"
        "- Als er geen geschillen waren, antwoord met een lege lijst: [].\n"
        "- Gebruik alleen rolnamen die in het debat voorkomen en record-ID's die echt genoemd zijn.\n"
        "- Gewoon koppelteken (-) in record-ID's, geen speciaal teken.\n\n"
        "Debat:\n\n" + "\n\n".join(debat)
    )
    geschillen_r = await koning.run(geschillen_prompt, options=opties)
    tel(verbruik, "groot", geschillen_r.usage_details)
    geschillen_ruw = (geschillen_r.text or "").strip().replace("‑", "-")
    g_start, g_eind = geschillen_ruw.find("["), geschillen_ruw.rfind("]")
    if g_start != -1 and g_eind != -1:
        try:
            trace["geschillen"] = json.loads(geschillen_ruw[g_start:g_eind + 1])
        except json.JSONDecodeError:
            trace["geschillen"] = []  # geen harde fout: het Verdict is belangrijker
    bewaar("trace.json", trace)
    print(f"\n[Koning] geschillen gevonden: {len(trace['geschillen'])}")

    # Stap 2: Het Verdict in JSON — expliciet per deel van de oorzaak, met bewijs of eerlijk null.
    verdict_prompt = (
            "Je bent de Koning van de Council of Achrona. Op basis van het volledige debat hieronder, "
            "je eerdere samenvatting en de gevonden geschillen stel je nu het officiële Verdict op.\n\n"
            "Antwoord ALLEEN met geldige JSON in exact dit formaat (geen tekst eromheen, geen markdown-codefences):\n"
            "{\n"
            '  "samenvatting": "<één alinea die de hele zaak samenvat, max 4 zinnen>",\n'
            '  "oorzaak": {\n'
            '    "ingreep":    {"bewering": "<wie deed wat, concreet, of null>",            "bewijs": ["<record-ID>", ...]},\n'
            '    "mechanisme": {"bewering": "<hoe werkt het technisch/magisch, of null>",   "bewijs": ["<record-ID>", ...]},\n'
            '    "verloop":    {"bewering": "<hoe ging het in de tijd, met data/volgorde, of null>", "bewijs": ["<record-ID>", ...]}\n'
            "  },\n"
            '  "acties": [{"actie": "<wat te doen>", "reden": "<waarom>"}, ...],\n'
            '  "risicos": [{"risico": "<wat kan misgaan>", "mitigatie": "<hoe beperken>"}, ...]\n'
            "}\n\n"
            "Belangrijke regels voor de inhoud:\n"
            "- Werk de DRIE delen van de oorzaak afzonderlijk uit: ingreep (de dader/handeling), "
            "mechanisme (hoe de schade ontstaat), verloop (de volgorde of tijdlijn).\n"
            "- Elk deel krijgt een concrete bewering met minstens één record-ID als bewijs, OF "
            "\"bewering\": null en \"bewijs\": [] als het debat onvoldoende zekerheid gaf. Nooit vaag.\n"
            "- Gebruik alleen record-ID's die een Raadslid in het debat echt noemde. Verzin er geen.\n"
            "- Gebruik een gewoon koppelteken (-) in record-ID's, geen speciaal teken.\n"
            "- Geef minstens één actie (geordend herstelplan) en minstens één risico met mitigatie.\n"
            "- Als geschillen onopgelost zijn, weeg dat mee in de zekerheid: liever \"bewering\": null dan gokken.\n\n"
            "Debat:\n\n" + "\n\n".join(debat)
            + "\n\nJouw eerdere samenvatting:\n\n" + r.text
            + "\n\nGevonden geschillen:\n\n" + json.dumps(trace["geschillen"], indent=2, ensure_ascii=False)
    )
    verdict_r = await koning.run(verdict_prompt, options=opties)
    tel(verbruik, "groot", verdict_r.usage_details)

    # Lees het JSON-antwoord. Modellen zetten soms tekst eromheen, dus knippen we naar { ... }.
    ruw = verdict_r.text.strip()
    ruw = ruw.replace("‑", "-")  # vervang speciaal koppelteken door een gewoon
    start, eind = ruw.find("{"), ruw.rfind("}")
    if start == -1 or eind == -1:
        sys.exit(f"Koning gaf geen JSON terug:\n{ruw}")
    try:
        verdict = json.loads(ruw[start:eind + 1])
    except json.JSONDecodeError as e:
        sys.exit(f"Koning gaf ongeldige JSON terug: {e}\n{ruw}")
    verdict["run_id"] = run_id  # run_id voegen we zelf toe, niet aan het model overlaten

    trace["status"], trace["aan_het_woord"] = "klaar", None
    bewaar("trace.json", trace)
    bewaar("verdict.json", verdict)

    print(f"\nKlaar: verdict.json en trace.json geschreven (run_id {run_id}).")
    for model, v in verbruik.items():
        print(f"Verbruik {model}: {v['beurten']} beurten, {v['invoer']} tokens in ({v['gecached']} gecached), {v['uitvoer']} tokens uit")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(run())


if __name__ == "__main__":
    main()
