"""
Rave Atlas, Berlin club registry.

A single authoritative table of Berlin clubs with their official website,
events/program URL, street address, and a scrape hint. This module is the
canonical source consumed by:

  - the knowledge base (berlin_clubs_az.md mirrors this data for RAG retrieval)
  - any future events scraper (iterate CLUBS, fetch each `events_url`)
  - the agent, indirectly, via retrieval

Data provenance:
  Addresses, official websites, and berlin.de detail pages were harvested
  from berlin.de/en/clubs/a-z (the Berlin Senate's official club registry)
  and each club's own site. berlin.de is the authoritative address source.

The `scrape` field records what was empirically observed when fetching each
site as plain HTML (markdown via a simple GET):

  - "http": server-rendered; events are readable with a plain HTTP fetch.
               No headless browser needed. Cheapest to scrape.
  - "browser": JS-rendered shell; the events list is populated client-side,
               so a headless browser (Playwright) is required to see content.
  - "ra": events live primarily on Resident Advisor; use the existing
               find_events RA GraphQL tool rather than scraping the site.
  - "unknown": not yet probed.

This field lets a scraper route each club to the cheapest method that works
(HTTP-first, browser only when necessary) instead of launching a browser for
every site.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClubEntry:
    """One Berlin club / venue and how to reach its event listings."""

    name: str
    address: str
    website: str | None # official site root, or None if none listed
    events_url: str | None # direct events/program page, if known
    berlin_de: str | None # authoritative berlin.de detail page
    instagram: str | None # @handle, if confirmed
    scrape: str # "http" | "browser" | "ra" | "unknown"
    note: str = "" # one-line description / caveat


# ---------------------------------------------------------------------------
# Core electronic-music venues (deep-profiled; website + address confirmed)
# ---------------------------------------------------------------------------

CLUBS: list[ClubEntry] = [
    ClubEntry(
        "://about blank", "Markgrafendamm 24c, 10245 Berlin",
        "https://aboutblank.li", "https://aboutblank.li",
        "https://www.berlin.de/en/clubs/8870883-4469452-about-blank.en.html",
        None, "http",
        "Queer, anti-fascist techno and experimental; large wooded garden near Ostkreuz.",
    ),
    ClubEntry(
        "Berghain", "Am Wriezener Bahnhof, 10243 Berlin",
        "https://www.berghain.berlin", "https://www.berghain.berlin/de/program/",
        "https://www.berlin.de/en/clubs/8871117-4469452-berghain.en.html",
        None, "http",
        "The flagship Berlin techno institution; Panorama Bar, Saule, Lab.oratory.",
    ),
    ClubEntry(
        "Tresor", "Koepenicker Str. 70, 10179 Berlin",
        "https://tresorberlin.com", "https://tresorberlin.com/club/events/",
        "https://www.berlin.de/en/clubs/8872188-4469452-tresor.en.html",
        None, "http",
        "Founding Berlin techno venue since 1991; the vault, Globus upstairs.",
    ),
    ClubEntry(
        "OHM", "Koepenicker Str. 70, 10179 Berlin",
        "https://ohmberlin.com", "https://ohmberlin.com/upcoming",
        "https://www.berlin.de/en/clubs/8871792-4469452-ohm.en.html",
        None, "http",
        "Experimental electronic in the old Kraftwerk battery room; Tresor complex.",
    ),
    ClubEntry(
        "KitKatClub", "Koepenicker Str. 76, 10179 Berlin",
        "https://kitkatclub.org", "https://kitkatclub.org",
        "https://www.berlin.de/en/clubs/8871648-4469452-kitkatclub.en.html",
        None, "http",
        "Fetish and sex-positive techno; CarneBall Bizarre Saturdays, strict dress code.",
    ),
    ClubEntry(
        "Kater Blau", "Holzmarktstr. 25, 10243 Berlin",
        "https://katerblau.de", "https://www.katerclub.de",
        "https://www.berlin.de/en/clubs/8871624-4469452-kater-blau.en.html",
        None, "http",
        "Bar25 lineage on the Spree; house, techno, confetti; FOREVER 25, Katergarten.",
    ),
    ClubEntry(
        "Sisyphos", "Hauptstr. 15, 10317 Berlin",
        "https://www.sisyphos-berlin.net", "https://www.sisyphos-berlin.net",
        "https://www.berlin.de/en/clubs/8872029-4469452-sisyphos.en.html",
        None, "browser",
        "Marathon weekend factory complex; Hammahalle, Wintergarten, Dampfer, garden. Lineups via sisy.fan.",
    ),
    ClubEntry(
        "Wilde Renate", "Alt-Stralau 70, 10245 Berlin",
        "https://www.renate.cc", "https://www.renate.cc",
        "https://www.berlin.de/en/clubs/8871981-4469452-wilde-renate.en.html",
        None, "http",
        "Apartment-maze party labyrinth; BLACK/GREEN/RED rooms + free garden open airs.",
    ),
    ClubEntry(
        "Heidegluehen", "Seestrasse 1, 13353 Berlin (registered); events near Beusselstr. 52.5381N 13.3282E",
        "https://heidegluehen.berlin", "https://heidegluehen.berlin/monatsvorschau/",
        "https://www.berlin.de/en/clubs/9689902-4469452-heidegluehen.en.html",
        None, "http",
        "Numbered marathon house events (noon Sat to Sun), 21+, ticket office closes 3h before end.",
    ),
    ClubEntry(
        "Fitzroy", "Holzmarktstr. 15-18, 10179 Berlin",
        "https://fitzroy-berlin.de", "https://fitzroy-berlin.de/events/",
        None, "@fitzroyclub", "http",
        "Techno, bass, club, dubstep at Holzmarkt; Almost Always, Low Ends & Explorers.",
    ),
    ClubEntry(
        "RSO.Berlin", "Schnellerstr. 137, 12439 Berlin",
        "https://rso.berlin", "https://rso.berlin",
        "https://www.berlin.de/en/clubs/8871915-4469452-rso-berlin.en.html",
        None, "browser",
        "Former Barenquell brewery in Schoeneweide; open-airs, Home Again festival, CTM.",
    ),
    ClubEntry(
        "Club der Visionaere", "Am Flutgraben 1, 12435 Berlin",
        "https://clubdervisionaere.com", "https://clubdervisionaere.com",
        "https://www.berlin.de/en/clubs/8871273-4469452-club-der-visionaere.en.html",
        None, "http",
        "Minimal house/techno on a canal-side wooden deck; intimate afterhours.",
    ),
    ClubEntry(
        "Hoppetosse", "Eichenstr. 4, 12435 Berlin",
        "https://hoppetosse.berlin", "https://hoppetosse.berlin",
        "https://www.berlin.de/en/clubs/8871498-4469452-hoppetosse.en.html",
        None, "http",
        "Parties on a moored ship; three decks, house and techno, Spree views.",
    ),
    ClubEntry(
        "Else", "An den Treptowers 10, 12435 Berlin",
        None, None,
        "https://www.berlin.de/en/clubs/8871339-4469452-else.en.html",
        None, "unknown",
        "Seasonal open-air dancefloor on the Spree banks; house and techno.",
    ),
    ClubEntry(
        "SO36", "Oranienstr. 190, 10999 Berlin",
        "https://www.so36.de", "https://www.so36.de/programm/",
        "https://www.berlin.de/en/clubs/8872059-4469452-so36.en.html",
        None, "http",
        "Legendary punk/alt venue since 1978; Gayhane Turkish queer night, concerts.",
    ),
    ClubEntry(
        "Gretchen", "Obentrautstr. 19-21, 10963 Berlin",
        "https://www.gretchen-club.de", "https://www.gretchen-club.de",
        "https://www.berlin.de/en/clubs/8871447-4469452-gretchen.en.html",
        None, "http",
        "Former Prussian cavalry stable; electronic, hip-hop, world music in Kreuzberg.",
    ),
    ClubEntry(
        "Ritter Butzke", "Ritterstr. 24-26, 10969 Berlin",
        "https://club.ritterbutzke.com", "https://club.ritterbutzke.com",
        "https://www.berlin.de/en/clubs/8871927-4469452-ritter-butzke.en.html",
        None, "http",
        "Former factory hall; techno and house with installations; Sunday GMF party.",
    ),
    ClubEntry(
        "Oxi", "Wiesenweg 1-4, 10365 Berlin",
        None, None,
        "https://www.berlin.de/en/clubs/8871834-4469452-oxi.en.html",
        None, "unknown",
        "Queer FLINTA space; electro/house/techno, open-air dancefloor, day-to-night marathons.",
    ),
    ClubEntry(
        "Panke", "Gerichtstr. 23, 13347 Berlin",
        "https://www.pankeculture.com", "https://www.pankeculture.com",
        "https://www.berlin.de/en/clubs/9689806-4469452-panke-berlin.en.html",
        None, "http",
        "Former factory cultural venue in Wedding; diverse music, concerts, workshops.",
    ),
    ClubEntry(
        "Humboldthain", "Hochstr. 46, 13357 Berlin",
        "https://humboldthain.com", "https://humboldthain.com",
        "https://www.berlin.de/en/clubs/8871513-4469452-humboldthain.en.html",
        None, "http",
        "Electronic sounds in a relaxed industrial setting near Humboldthain park.",
    ),
    ClubEntry(
        "Golden Gate", "Schicklerstr. 4, 10179 Berlin",
        "https://goldengate-berlin.de", "https://goldengate-berlin.de",
        "https://www.berlin.de/en/clubs/8871423-4469452-golden-gate.en.html",
        None, "http",
        "Raw, loud, uncompromising techno in a small club beneath S-Bahn arches.",
    ),
    ClubEntry(
        "Crack Bellmer", "Revaler Str. 99, 10245 Berlin (RAW-Gelaende)",
        "https://www.crackbellmer.de", "https://www.crackbellmer.de/program/this-month",
        "https://www.berlin.de/en/clubs/9444202-4469452-crack-bellmer.en.html",
        None, "http",
        "House, techno and experimental formats; bar-club on the RAW-Gelaende.",
    ),
    ClubEntry(
        "Prince Charles", "Prinzenstr. 85f, 10969 Berlin",
        "https://www.princecharlesberlin.com", "https://www.princecharlesberlin.com",
        "https://www.berlin.de/en/clubs/8871858-4469452-prince-charles.en.html",
        None, "http",
        "House, electro, hip-hop in a former swimming pool with a pool-shaped floor.",
    ),
    ClubEntry(
        "Birgit (und Bier)", "Schleusenufer 3, 10997 Berlin",
        "https://www.birgit.club", "https://www.birgit.club",
        "https://www.berlin.de/en/clubs/8871141-4469452-birgit.en.html",
        None, "http",
        "Open-air club and beer-garden on the canal; festival vibe, SONAR safer-nightlife host.",
    ),
    ClubEntry(
        "Weekend Club", "Alexanderstr. 7, 10178 Berlin",
        "https://www.weekendclub.berlin", "https://www.weekendclub.berlin/upcoming",
        "https://www.berlin.de/en/clubs/8872245-4469452-weekend-club.en.html",
        None, "http",
        "Rooftop and main floor with TV-tower views; techno, hip-hop, Afro, Latin.",
    ),
    ClubEntry(
        "Arkaoda", "Karl-Marx-Platz 16, 12043 Berlin",
        "https://berlin.arkaoda.com", "https://berlin.arkaoda.com/?/default/program",
        "https://www.berlin.de/en/clubs/9659830-4469452-arkaoda.en.html",
        None, "http",
        "Post-punk, krautrock, dub, ambient, disco, electronic; Neukoelln cultural club.",
    ),

    # -----------------------------------------------------------------------
    # RAW-Gelaende cluster (all at Revaler Str. 99, multi-venue site)
    # -----------------------------------------------------------------------
    ClubEntry(
        "Cassiopeia", "Revaler Str. 99, 10245 Berlin (RAW-Gelaende)",
        "https://www.cassiopeia-berlin.de", "https://www.cassiopeia-berlin.de",
        "https://www.berlin.de/en/clubs/8871201-4469452-cassiopeia.en.html",
        None, "http",
        "Multi-floor subculture venue in a former industrial hall; varied programming.",
    ),
    ClubEntry(
        "Badehaus", "Revaler Str. 99, 10245 Berlin (RAW-Gelaende)",
        "https://badehaus-berlin.com", "https://badehaus-berlin.com",
        "https://www.berlin.de/en/clubs/8871066-4469452-badehaus.en.html",
        None, "http",
        "Converted bathhouse; live acts and varied club nights, raw atmosphere.",
    ),
    ClubEntry(
        "Astra Kulturhaus", "Revaler Str. 99, 10245 Berlin (RAW-Gelaende)",
        "https://www.astra-berlin.de", "https://www.astra-berlin.de",
        "https://www.berlin.de/en/clubs/8871015-4469452-astra-kulturhaus.en.html",
        None, "http",
        "Big-stage concert and club venue; large capacity, regular party series.",
    ),
    ClubEntry(
        "Lokschuppen", "Revaler Str. 99, 10245 Berlin (RAW-Gelaende)",
        "https://lokschuppen-berlin.com", "https://lokschuppen-berlin.com/2/",
        "https://www.berlin.de/en/clubs/8872104-4469452-lokschuppen.en.html",
        None, "browser",
        "Driving-techno nights in a former locomotive shed on the RAW-Gelaende.",
    ),

    # -----------------------------------------------------------------------
    # Larger / mixed-genre and queer venues
    # -----------------------------------------------------------------------
    ClubEntry(
        "Metropol", "Nollendorfplatz 5, 10777 Berlin",
        "https://metropol-berlin.de", "https://metropol-berlin.de/events",
        "https://www.berlin.de/en/clubs/9703630-4469452-metropol-berlin.en.html",
        None, "http",
        "Historic 1,000-cap theatre on Nollendorfplatz; SchwuZ relaunch parties from 2026.",
    ),
    ClubEntry(
        "SchwuZ", "Rollbergstr. 26, 12053 Berlin (CLOSED Nov 2025; relaunching at Metropol)",
        None, None,
        "https://www.berlin.de/en/clubs/8871999-4469452-schwuz.en.html",
        None, "unknown",
        "Germany's oldest/largest queer club; insolvency Aug 2025, Neukoelln site closed Nov 2025.",
    ),
    ClubEntry(
        "Lido", "Cuvrystr. 7, 10997 Berlin",
        "https://www.lido-berlin.de", "https://www.lido-berlin.de",
        "https://www.berlin.de/en/clubs/8871690-4469452-lido.en.html",
        None, "http",
        "Indie, rock, pop in a historic Kreuzberg cinema-club; live concerts, themed parties.",
    ),
    ClubEntry(
        "Yaam", "An der Schillingbruecke 3, 10243 Berlin",
        "https://www.yaam.de", "https://www.yaam.de",
        "https://www.berlin.de/en/clubs/8872275-4469452-yaam.en.html",
        None, "http",
        "Afro-Caribbean beach club on the Spree; afrobeat, reggae, dancehall, hip-hop.",
    ),
    ClubEntry(
        "Monarch", "Skalitzer Str. 134, 10999 Berlin",
        "https://kottimonarch.de", "https://kottimonarch.de",
        "https://www.berlin.de/en/clubs/9689647-4469452-monarch.en.html",
        None, "http",
        "Living-room-flair bar-club above Kottbusser Tor; techno to funky pop.",
    ),
    ClubEntry(
        "Bi Nuu", "Schlesisches Tor (Skalitzer Str. 134), 10997 Berlin",
        "https://www.binuu.de", "https://www.binuu.de",
        "https://www.berlin.de/en/clubs/8871129-4469452-bi-nuu.en.html",
        None, "http",
        "Wide range from hardcore-punk concerts to electro parties at Schlesisches Tor.",
    ),
]


# ---------------------------------------------------------------------------
# Remaining registered clubs (berlin.de authoritative link; not deep-probed).
# Kept lighter on purpose, they are mainstream, niche, or boutique venues
# outside the app's core electronic focus, but included for completeness so
# the agent can still surface an official link if asked.
# ---------------------------------------------------------------------------

OTHER_CLUBS: list[ClubEntry] = [
    ClubEntry(
        "Acud Macht Neu", "Veteranenstr. 21, 10119 Berlin",
        "https://acudmachtneu.de", "https://acudmachtneu.de",
        "https://www.berlin.de/en/clubs/8871027-4469452-acud-macht-neu.en.html",
        None, "http",
        "Small alternative cultural venue; art and music programming in Mitte.",
    ),
    ClubEntry(
        "Aeden", "Schleusenufer 3, 10997 Berlin",
        "https://aedenberlin.com", None,
        "https://www.berlin.de/en/clubs/8870937-4469452-aeden.en.html",
        None, "unknown",
        "Boutique venue on the canal, Kreuzberg.",
    ),
    ClubEntry(
        "Alte Kantine", "Knaackstr. 97, 10435 Berlin",
        "https://alte-kantine.eu", None,
        "https://www.berlin.de/en/clubs/8870964-4469452-alte-kantine.en.html",
        None, "unknown",
        "Former factory canteen in Kulturbrauerei, Prenzlauer Berg.",
    ),
    ClubEntry(
        "AM Club", "Brunsbuetteler Damm 51-53, 13581 Berlin",
        None, None,
        "https://www.berlin.de/en/clubs/10056826-4469452-am-club-berlin.en.html",
        None, "unknown",
        "Boutique electronic venue in western Berlin.",
    ),
    ClubEntry(
        "Ballhaus Spandau", "Dorfstr. 5, 13597 Berlin",
        "https://ballhaus-spandau.club", None,
        "https://www.berlin.de/en/clubs/8871084-4469452-ballhaus-spandau.en.html",
        None, "unknown",
        "Dance hall, Spandau.",
    ),
    ClubEntry(
        "Beate Uwe", "Schillingstr. 31, 10179 Berlin",
        "https://beate-uwe.de", "https://beate-uwe.de",
        "https://www.berlin.de/en/clubs/8871099-4469452-beate-uwe.en.html",
        None, "http",
        "Small alternative venue near Alexanderplatz; Fri/Sat/Sun evenings.",
    ),
    ClubEntry(
        "Bohnengold", "Reichenberger Str. 153, 10999 Berlin",
        "https://bohnengold.de", None,
        "https://www.berlin.de/en/clubs/8871156-4469452-bohnengold.en.html",
        None, "unknown",
        "Small boutique venue in Kreuzberg.",
    ),
    ClubEntry(
        "Bricks", "Mohrenstr. 30, 10117 Berlin",
        "https://bricks-berlin.club", None,
        "https://www.berlin.de/en/clubs/8871171-4469452-bricks.en.html",
        None, "unknown",
        "Central club, mixed electronic, Mitte.",
    ),
    ClubEntry(
        "Bulbul", "Skalitzer Str. 114, 10999 Berlin",
        None, None,
        "https://www.berlin.de/en/clubs/8871183-4469452-bulbul.en.html",
        None, "ra",
        "Small venue; RA is the primary listing.",
    ),
    ClubEntry(
        "Busche Club", "Warschauer Platz 18, 10245 Berlin",
        None, None,
        "https://www.berlin.de/en/clubs/8871192-4469452-busche-club.en.html",
        None, "unknown",
        "PERMANENTLY CLOSED as of July 2025.",
    ),
    ClubEntry(
        "Der Weisse Hase", "Revaler Str. 99, 10245 Berlin (RAW-Gelaende)",
        "https://derweissehase.club", None,
        "https://www.berlin.de/en/clubs/8871318-4469452-der-weisse-hase.en.html",
        None, "unknown",
        "Small Friedrichshain venue on the RAW-Gelaende.",
    ),
    ClubEntry(
        "Duncker", "Dunckerstr. 64, 10439 Berlin",
        "https://dunckerclub.de", None,
        "https://www.berlin.de/en/clubs/8871327-4469452-duncker.en.html",
        None, "unknown",
        "Prenzlauer Berg; gothic and alternative nights.",
    ),
    ClubEntry(
        "Frannz", "Schoenhauser Allee 36, 10435 Berlin",
        "https://frannz.eu", None,
        "https://www.berlin.de/en/clubs/8871399-4469452-frannz.en.html",
        None, "unknown",
        "Medium venue in Kulturbrauerei, Prenzlauer Berg.",
    ),
    ClubEntry(
        "Hafenbar", "Karl-Liebknecht-Str. 11, 10178 Berlin",
        "https://hafenbar-berlin.de", None,
        "https://www.berlin.de/en/clubs/8871459-4469452-hafenbar.en.html",
        None, "unknown",
        "Established bar-club in Mitte.",
    ),
    ClubEntry(
        "Havanna", "Hauptstr. 30, 10827 Berlin",
        "https://havanna-berlin.de", None,
        "https://www.berlin.de/en/clubs/8871486-4469452-havanna-berlin.en.html",
        None, "unknown",
        "Salsa and Latin dance club, Schoeneberg.",
    ),
    ClubEntry(
        "Insomnia", "Alt Tempelhof 17-19, 12099 Berlin",
        "https://insomnia-berlin.de", None,
        "https://www.berlin.de/en/clubs/8871528-4469452-insomnia.en.html",
        None, "unknown",
        "Erotic/swinger-oriented club near Tempelhof.",
    ),
    ClubEntry(
        "L.U.X.", "Schlesische Str. 41, 10997 Berlin",
        "https://lux-berlin.net", None,
        "https://www.berlin.de/en/clubs/8871660-4469452-lux.en.html",
        None, "unknown",
        "Boutique venue in Kreuzberg.",
    ),
    ClubEntry(
        "M01", "Markgrafendamm 1, 10245 Berlin",
        None, None,
        "https://www.berlin.de/en/clubs/9703495-4469452-m01-berlin.en.html",
        None, "unknown",
        "Medium venue near Ostkreuz; varied programming.",
    ),
    ClubEntry(
        "Matrix Club", "Warschauer Platz 18, 10245 Berlin",
        "https://matrix-berlin.de", "https://matrix-berlin.de",
        "https://www.berlin.de/en/clubs/8871732-4469452-matrix-club.en.html",
        None, "http",
        "Large commercial club under S-Bahn arches, Warschauer Str.; open daily.",
    ),
    ClubEntry(
        "Maxxim Club", "Joachimstaler Str. 15, 10719 Berlin",
        "https://maxxim-berlin.de", None,
        "https://www.berlin.de/en/clubs/8871747-4469452-maxxim-club.en.html",
        None, "unknown",
        "Upscale commercial club in Charlottenburg; open 365 days a year.",
    ),
    ClubEntry(
        "M-bia", "Dircksenstr. 123, 10178 Berlin",
        "https://m-bia.de", None,
        "https://www.berlin.de/en/clubs/8871720-4469452-mbia.en.html",
        None, "unknown",
        "Established Kreuzberg basement venue.",
    ),
    ClubEntry(
        "Ost", "Alt-Stralau 1, 10245 Berlin",
        "https://clubost.de", None,
        "https://www.berlin.de/en/clubs/8871804-4469452-ost.en.html",
        None, "unknown",
        "Eastern Berlin waterfront venue.",
    ),
    ClubEntry(
        "Paloma Bar", "Skalitzer Str. 135, 10999 Berlin",
        "https://palomabar.de", None,
        "https://www.berlin.de/en/clubs/9689695-4469452-paloma-bar.en.html",
        None, "unknown",
        "Small dancefloor and bar above Kottbusser Tor.",
    ),
    ClubEntry(
        "PrivatClub", "Skalitzer Str. 85-86, 10997 Berlin",
        "https://privatclub-berlin.de", None,
        "https://www.berlin.de/en/clubs/8871888-4469452-privatclub.en.html",
        None, "unknown",
        "Basement club; concerts and parties, Kreuzberg.",
    ),
    ClubEntry(
        "Silverwings Club", "Columbiadamm 10f, 10965 Berlin",
        None, None,
        "https://www.berlin.de/en/clubs/8872011-4469452-silverwings-club.en.html",
        None, "unknown",
        "Venue near former Tempelhof airport; Facebook-only listing.",
    ),
    ClubEntry(
        "Soda Club", "Schoenhauser Allee 36, 10435 Berlin",
        "https://soda-berlin.de", None,
        "https://www.berlin.de/en/clubs/8872071-4469452-soda-club.en.html",
        None, "unknown",
        "Kulturbrauerei; salsa and mainstream club nights, Prenzlauer Berg.",
    ),
    ClubEntry(
        "Soulcat", "Pannierstr. 53, 12047 Berlin",
        "https://soulcat-berlin.com", None,
        "https://www.berlin.de/en/clubs/8872080-4469452-soulcat.en.html",
        None, "unknown",
        "Smaller boutique venue in Neukoelln.",
    ),
    ClubEntry(
        "Surprise Club", "Potsdamer Str. 84, 10785 Berlin",
        "https://surprise-berlin.de", None,
        "https://www.berlin.de/en/clubs/8872119-4469452-surprise-club.en.html",
        None, "unknown",
        "Boutique club in Tiergarten.",
    ),
    ClubEntry(
        "The Pearl", "Fasanenstr. 81, 10623 Berlin",
        "https://thepearl-berlin.de", None,
        "https://www.berlin.de/en/clubs/8872137-4469452-the-pearl.en.html",
        None, "unknown",
        "Upscale commercial club in Charlottenburg.",
    ),
    ClubEntry(
        "Trompete", "Luetzowplatz 9, 10785 Berlin",
        "https://trompete-berlin.de", None,
        "https://www.berlin.de/en/clubs/2225559-4469452-trompete.en.html",
        None, "unknown",
        "Small Tiergarten venue; Thursdays from 9pm, 3 nights a week.",
    ),
    ClubEntry(
        "Void Club", "Wiesenweg 5-9, 10365 Berlin",
        "https://void-club.de", None,
        "https://www.berlin.de/en/clubs/8872209-4469452-void-club.en.html",
        None, "unknown",
        "Small underground techno venue.",
    ),
    ClubEntry(
        "Zita", "Am Juliusturm 64, 13599 Berlin",
        None, None,
        "https://www.berlin.de/en/clubs/9703522-4469452-zita-berlin.en.html",
        None, "unknown",
        "Cinema-club hybrid in Spandau.",
    ),
    ClubEntry(
        "Zur Klappe", "Yorckstr. 2, 10965 Berlin",
        "https://zurklappe.org", None,
        "https://www.berlin.de/en/clubs/8872302-4469452-zur-klappe.en.html",
        None, "unknown",
        "Historic gay cruising-bar in Kreuzberg.",
    ),
    ClubEntry(
        "808 Berlin", "Budapester Str. 38-40, 10787 Berlin",
        "https://808.berlin", None,
        "https://www.berlin.de/en/clubs/8870916-4469452-808-berlin.en.html",
        None, "unknown",
        "Small electronic venue in Charlottenburg.",
    ),
]


ALL_CLUBS: list[ClubEntry] = CLUBS + OTHER_CLUBS


def get_club(name: str) -> ClubEntry | None:
    """Case-insensitive lookup by club name."""
    needle = name.strip().lower()
    for c in ALL_CLUBS:
        if c.name.lower() == needle:
            return c
    # loose contains-match fallback
    for c in ALL_CLUBS:
        if needle in c.name.lower():
            return c
    return None


def clubs_with_events_url() -> list[ClubEntry]:
    """Clubs that have a known events/program page (scraper candidates)."""
    return [c for c in ALL_CLUBS if c.events_url]


def clubs_by_scrape_method(method: str) -> list[ClubEntry]:
    """All clubs whose events page is reachable by the given method."""
    return [c for c in ALL_CLUBS if c.scrape == method]


if __name__ == "__main__":
    print(f"Total clubs registered : {len(ALL_CLUBS)}")
    print(f" deep-profiled : {len(CLUBS)}")
    print(f" link-only : {len(OTHER_CLUBS)}")
    print(f" with events_url : {len(clubs_with_events_url())}")
    print()
    for m in ("http", "browser", "ra", "unknown"):
        clubs = clubs_by_scrape_method(m)
        print(f" scrape={m:8s}: {len(clubs)}")
    print()
    # sanity: every entry has a resolvable link of some kind
    for c in ALL_CLUBS:
        assert c.website or c.berlin_de, f"{c.name} has no link at all"
    print("All clubs have at least one official link. OK.")
    real_websites = sum(1 for c in OTHER_CLUBS if c.website)
    print(f" other with real website: {real_websites}/{len(OTHER_CLUBS)}")
