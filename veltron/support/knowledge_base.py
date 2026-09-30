"""Product knowledge base for the customer-support system.

IMPORTANT: every document below is **synthetic demo content authored for this project**.
Veltron Industries does not exist. The policies here are invented to exercise the
retrieval, grounding and escalation paths, and must never be presented as a real
company's terms. Each document carries ``synthetic: true`` and the loader stamps that
into every retrieved chunk so the API can surface it to operators.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.logging_utils import get_logger

log = get_logger(__name__)

KB_ROOT = Path(__file__).resolve().parents[2] / "knowledge_base"

DEMO_DISCLAIMER = (
    "SYNTHETIC DEMO DATA. Veltron Industries is a fictional company created for "
    "engineering demonstration. These policies are invented and are not legally binding."
)


# --------------------------------------------------------------------------- products
PRODUCTS: list[dict[str, Any]] = [
    {
        "doc_id": "product-veltron-hub",
        "category": "products",
        "title": "VeltronHub X1 smart hub",
        "language": "en",
        "tags": ["hardware", "hub", "x1", "connectivity", "specifications"],
        "body": """
# VeltronHub X1

The VeltronHub X1 is Veltron Industries' central smart-home controller. It ships with
the VeltronHub app (iOS 15+, Android 11+) and requires a Veltron Account to activate.

## Specifications
- Model number: VH-X1-EU
- Network: dual-band Wi-Fi 6 (802.11ax) and Gigabit Ethernet
- Protocols: Matter 1.2, Zigbee 3.0, Z-Wave 800
- Bluetooth: 5.3 (commissioning only)
- Power: 12V DC, 30W maximum, USB-C PD input accepted
- Operating temperature: 0 C to 40 C
- Dimensions: 118 x 78 x 26 mm, 240 g
- Warranty: 24 months from the date shown on the purchase receipt

## What is in the box
1x VeltronHub X1, 1x 12V/30W power adapter, 1x 1.5 m Gigabit Ethernet cable,
1x quick-start guide, 1x mounting template.

## Activation
An internet connection is required. Open the Veltron app, choose "Add device", and hold
the pairing button for 5 seconds until the status LED pulses amber. If the LED does not
pulse amber within 30 seconds, the device is not in pairing mode.
        """,
    },
    {
        "doc_id": "product-veltron-sense",
        "category": "products",
        "title": "VeltronSense environmental sensor",
        "language": "en",
        "tags": ["sensor", "temperature", "humidity", "air quality", "battery"],
        "body": """
# VeltronSense

VeltronSense measures temperature, relative humidity, VOC and particulate matter, and
repeats those readings to VeltronHub X1 over Zigbee.

## Specifications
- Model number: VS-2-EU
- Sensors: temperature +/-0.2 C, humidity +/-2% RH, VOC (eCO2 proxy), PM2.5
- Radio: Zigbee 3.0, 20 m line of sight
- Battery: 2 x AA, expected 18 months at 1 reading per minute
- Operating temperature: -10 C to 50 C

## Known limitation
VeltronSense does not report CO2 concentration directly. The "CO2" figure in the app is an
inferred equivalent derived from VOC and is labelled "eCO2 inferred" in the interface.
Support agents must not describe this number as a direct CO2 measurement.
        """,
    },
    {
        "doc_id": "product-veltron-cloud",
        "category": "products",
        "title": "Veltron Cloud subscription tiers",
        "language": "en",
        "tags": ["subscription", "cloud", "plan", "tier", "pricing", "storage"],
        "body": """
# Veltron Cloud plans

Veltron Cloud is optional. The VeltronHub X1 functions fully on the local network without
a subscription; cloud features add remote access, history retention and multi-site control.

| Plan | Monthly price (EUR) | History retention | Sites | Remote access |
|------|--------------------|-------------------|-------|---------------|
| Local (free) | 0.00 | 7 days, local only | 1 | No |
| Plus | 4.99 | 24 months | 3 | Yes |
| Pro | 11.99 | 60 months | 15 | Yes |

Annual billing is discounted to 10 months for 12 months (Plus 59.88 EUR/year,
Pro 143.88 EUR/year). Prices are list prices in EUR excluding VAT. Regional pricing may
differ and is shown at checkout before confirmation.

## Cancellation
Subscriptions renew automatically every billing period. Cancellation takes effect at the
end of the current paid period; no partial refunds are issued for the remaining days of
an already-paid period, except where local consumer law requires it.
        """,
    },
]

# -------------------------------------------------------------------------- policies
POLICIES: list[dict[str, Any]] = [
    {
        "doc_id": "policy-warranty",
        "category": "policies",
        "title": "Warranty and RMA policy",
        "language": "en",
        "tags": ["warranty", "rma", "return", "defect", "claim", "repair", "replacement"],
        "body": """
# Warranty and RMA

## Coverage
Veltron hardware carries a 24-month limited warranty from the purchase date. This covers
manufacturing defects in materials and workmanship under normal use.

## Not covered
- Physical damage, liquid ingress, or use outside the documented operating range
- Consumables: batteries, silicone sleeves, mounting adhesive
- Damage caused by third-party modifications or by powering from an unlisted adapter
- Software configuration issues, which are resolved by configuration, not replacement

## Making a claim
1. Open a ticket in the Veltron app or on the support site and select "Hardware defect".
2. Provide the device serial number and the purchase receipt (receipt number or the order
   email). A photo of the fault is helpful but not mandatory.
3. Veltron Support issues an RMA number within 2 business days.
4. Ship the device with the RMA number written on the outside of the parcel.

## Turnaround
- Assessment: 5 business days after the return arrives at the Veltron Depot in Novi Sad.
- Replacement or repair: 10 business days after assessment.
- The customer pays return shipping for the first claim on a device.
- Veltron pays return shipping for claims on the second and later claim for the same device
  within 24 months.

## Out-of-warranty service
Repairs are quoted before work begins and are never performed without written approval.
Quoted price: 45.00 EUR diagnostic fee, credited against the repair if approved; repair
labour 60.00 EUR per hour, parts at cost plus 15% handling.
        """,
    },
    {
        "doc_id": "policy-refund",
        "category": "policies",
        "title": "Refund policy",
        "language": "en",
        "tags": ["refund", "return", "money", "chargeback", "cooling off"],
        "body": """
# Refund policy

## Standard returns
Physical products may be returned within 30 days of delivery in original packaging with
all accessories. The customer pays return shipping. Refunds are issued to the original
payment method within 14 days of the return being received and inspected.

## Digital subscriptions
Subscription refunds follow the cancellation terms in the Veltron Cloud plan document.
Refunds for a partially used billing period are not issued, except where local consumer
law provides otherwise. In the European Union this right applies for 14 days from purchase
for digital content that has not been fully consumed.

## Chargebacks
Disputed card payments should be raised with Veltron Support first so the order can be
reviewed. Chargebacks opened without contacting support delay the investigation and do
not guarantee a refund.

## Non-refundable situations
- Products damaged after delivery
- Digital services already consumed in full
- Custom configuration services already performed
- Items bought from an unauthorised reseller, which are handled by that reseller
        """,
    },
    {
        "doc_id": "policy-shipping",
        "category": "shipping",
        "title": "Shipping and delivery",
        "language": "en",
        "tags": ["shipping", "delivery", "dispatch", "tracking", "carrier", "delay"],
        "body": """
# Shipping

## Dispatch times
| Region | Dispatch |
|--------|----------|
| Serbia | 1 business day |
| EU | 2 business days |
| UK and Norway | 3 business days |
| Rest of world | 5 business days |

Dispatch is in business days, excluding weekends and public holidays in Serbia. Orders
placed after 15:00 CET are treated as next business day.

## Carriers
Serbia: Post Express. EU: DHL Express. UK and Norway: Bring. Rest of world: DHL Express.

## Tracking
A tracking reference is emailed when the parcel is handed to the carrier. Tracking may
take up to 24 hours to show the first scan. If tracking has shown no movement for 5
business days, contact support.

## Delays and loss
If a parcel is lost, support opens a carrier investigation. Carrier investigation windows
are typically 10 business days for DHL and 14 business days for Post Express. Veltron
replaces a confirmed-lost device in hardware or issues a full refund of the product
price; shipping charges are not refunded.
        """,
    },
    {
        "doc_id": "policy-privacy-data",
        "category": "policies",
        "title": "Privacy and data handling",
        "language": "en",
        "tags": ["privacy", "data", "gdpr", "personal data", "retention", "export"],
        "body": """
# Privacy and data handling

## What the Veltron Cloud stores
- Account identifiers: name, email address, Veltron Account ID
- Device metadata: serial number, model, firmware version, pairing date
- Telemetry: sensor readings at the interval configured on the device (default 1 minute)
- Support tickets: message text, attachments you choose to include, agent replies

## Retention
- Telemetry: the retention period of your plan (7 days local-only, 24 months Plus, 60 months Pro)
- Support tickets: 24 months after the ticket is closed
- Account data: deleted within 30 days of account closure

## Your rights
Under the GDPR you may request access, correction, export, erasure, restriction, and
objection. Requests are answered within 30 days. Export produces a JSON archive and a CSV
of sensor history within 7 days of the request being accepted.

Support agents cannot delete an account on request received only through the app. Account
closure must be submitted from account settings, or by email to privacy@veltron.example,
so that identity can be verified.

## Sharing
Telemetry is not sold and is not shared with advertising networks. Aggregate, non-
reversible statistics are shared with manufacturing to forecast demand.
        """,
    },
    {
        "doc_id": "policy-security",
        "category": "policies",
        "title": "Account security",
        "language": "en",
        "tags": ["security", "password", "2fa", "account", "lockout", "hijack"],
        "body": """
# Account security

## Passwords
Minimum 12 characters. Passwords are not recoverable by support; support cannot read,
reset, or confirm a customer's password. A password reset link is sent by email.

## Two-factor authentication
Available on all plans. Agents must never ask a customer to read a 2FA code aloud or to
type it into a chat. A legitimate Veltron support agent will never request a password,
a 2FA code, a full card number, or a complete payment credential.

## Lockout
Five failed attempts locks the account for 30 minutes. Attempts continue to accumulate
after unlock. Ten failures in 24 hours lock the account for 24 hours.

## Suspicious activity
If a customer reports they did not authorise a login or a pairing, treat it as a security
incident: reset the password immediately, remove all active sessions, and escalate to the
security on-call engineer. Do not attempt to diagnose the intrusion further at tier 1.
        """,
    },
]

# ---------------------------------------------------------------------- troubleshooting
TROUBLESHOOTING: list[dict[str, Any]] = [
    {
        "doc_id": "trouble-hub-offline",
        "category": "troubleshooting",
        "title": "VeltronHub X1 shows as offline",
        "language": "en",
        "tags": ["offline", "hub", "connectivity", "wifi", "ethernet", "network"],
        "body": """
# VeltronHub X1 shows as offline

Work through these in order. Each step has a verification, so do not skip ahead.

1. **Power.** Confirm the power LED is solid white. A pulsing amber LED means pairing mode;
   a blinking red LED means a boot failure -- continue to step 4.
2. **Ethernet.** If a cable is plugged in, confirm the link LED next to the port is lit.
   Swap the cable and the port before continuing.
3. **Wi-Fi band.** VeltronHub X1 supports 2.4 GHz and 5 GHz only. It does not join networks
   on 5 GHz channel 160 MHz, nor Wi-Fi 6E-only networks. Separate the 2.4 GHz and 5 GHz
   SSIDs if the router combines them.
4. **Firmware.** Power-cycle: hold the power button 10 seconds. If the LED returns to solid
   white but the device is still offline, install the firmware from
   Settings > Device > Firmware > Install. Firmware installs need 15 minutes and must not be
   interrupted.
5. **Account region.** The hub must be paired to an account in the same region as the store
   it was bought from. A hub bought in the EU cannot be claimed by an account registered in
   a different region. Region is set at purchase and cannot be changed by support.
6. **Escalate** if the LED is blinking red after a power-cycle and the firmware install
   completes, or if the hub has never been paired. These indicate a hardware fault and go
   to tier 2 under the warranty and RMA policy.
        """,
    },
    {
        "doc_id": "trouble-sensor-dropout",
        "category": "troubleshooting",
        "title": "VeltronSense drops out intermittently",
        "language": "en",
        "tags": ["sensor", "zigbee", "dropout", "battery", "range", "interference"],
        "body": """
# VeltronSense drops out intermittently

1. **Battery.** Replace both AA cells even if the app shows 60% or higher; alkaline cells
   with a partial load sag and cause resets. Use the same brand in both slots.
2. **Range.** Zigbee range is 20 m line of sight. Walls containing reinforced concrete or
   metal reduce it sharply. Move the sensor within the same room as a hub or add a Zigbee
   repeater; Veltron sells the VR-1 repeater.
3. **Wi-Fi interference.** 2.4 GHz Wi-Fi and Zigbee share spectrum. If the hub is on 2.4 GHz
   and the router is on channel 6, move the router to channel 11 or put the hub on Ethernet.
4. **Firmware.** Update both the hub and the sensor. Sensor firmware is pushed through the hub.
5. **Report.** If dropouts persist after all four, open a ticket with the hub serial, the
   sensor serial, and the times of at least three dropouts. Without timestamps tier 2 cannot
   correlate the events and will ask for them again.
        """,
    },
    {
        "doc_id": "trouble-app-pairing",
        "category": "troubleshooting",
        "title": "App cannot find the hub during pairing",
        "language": "en",
        "tags": ["pairing", "app", "bluetooth", "permission", "ios", "android"],
        "body": """
# App cannot find the hub during pairing

1. **Permissions.** Pairing uses Bluetooth for discovery only. On iOS the app must have
   Bluetooth and Local Network permission; on Android, Nearby Devices and Location
   permission (Android derives Nearby Devices from location). A denied permission produces
   an empty device list with no error message.
2. **Distance.** Pairing discovery works within about 4 m. Move the phone and reduce the
   number of active Bluetooth peripherals, particularly on shared office desktops.
3. **Bluetooth state.** Turn Bluetooth off and on again, then reopen the app. The pairing
   session does not survive a Bluetooth reset.
4. **Account signed in.** The app must already be signed in before "Add device" is offered.
   Signing in mid-flow restarts pairing.
5. **One hub per account.** An account may claim only one VeltronHub X1. If the hub is
   already claimed, the app will not list it; the existing owner must remove it under
   Settings > Device > Remove, or support must perform a transfer (see the account transfer
   document).
        """,
    },
    {
        "doc_id": "trouble-subscription",
        "category": "billing",
        "title": "Subscription charge and invoice questions",
        "language": "en",
        "tags": ["billing", "subscription", "invoice", "vat", "charge", "cancel"],
        "body": """
# Subscription charge and invoice questions

## Why was I charged?
Subscriptions renew automatically at the end of each paid period. The app sends a reminder
3 days before renewal. The renewal charge appears on the statement with the descriptor
VELTRON CLOUD and the last 4 digits of the card on file.

## I do not recognise the charge
Do not share card details in a ticket. Ask the customer to compare the last 4 digits and
the amount against their own records for the billing date. If the charge is legitimate but
unexpected, the renewal reminder email may have been filtered. Offer cancellation and
refund review per the refund policy.

## How do I get an invoice?
Invoices are issued automatically on the billing date and emailed to the account billing
address. To reissue: Settings > Billing > Invoice history > Reissue. A reissued invoice
arrives within 2 business days. Support can reissue an invoice but cannot change the
billing date, the VAT rate, or the legal entity on a past invoice; those require the
account's country of registration to be correct, which can be changed in
Settings > Account > Region.

## VAT
Prices in the plan table exclude VAT. The rate applied is the rate in the account's region
at the time of purchase. Changing the account region does not retroactively change VAT on
invoices already issued.
        """,
    },
    {
        "doc_id": "trouble-account-transfer",
        "category": "policies",
        "title": "Device ownership transfer",
        "language": "en",
        "tags": ["transfer", "ownership", "second hand", "claim", "region"],
        "body": """
# Device ownership transfer

A VeltronHub X1 can be claimed by a new account. Transfer requires proof of purchase.

## Required evidence
Either the original order email or the receipt showing the serial number and purchase date.

## Process
1. Support opens an "Ownership transfer" ticket.
2. The claimant supplies proof of purchase.
3. The existing owner is asked to release the device, or, if unreachable for 14 days,
   Support may release the device on the strength of the proof of purchase alone. A
   release on proof alone is logged and the serial is recorded against both accounts.
4. Regional pairing is reset. The new owner must set the region in the store they bought
   the device in; region cannot be changed afterwards.

Support must not release a device without proof of purchase, and must not accept a
photograph of a serial number alone.
        """,
    },
]

# ------------------------------------------------------------------------- FAQ
FAQ: list[dict[str, Any]] = [
    {
        "doc_id": "faq-general",
        "category": "faq",
        "title": "Frequently asked questions",
        "language": "en",
        "tags": ["faq", "general", "common questions"],
        "body": """
# FAQ

**Does the hub work without an internet connection?**
Yes, local control, automations and sensors all work on the local network. Remote access
and history beyond 7 days require a Veltron Cloud plan.

**Can I move my subscription between accounts?**
No. A plan belongs to the account that bought it. If both accounts are yours, cancel on one
and start a new plan on the other; the history on the old account does not transfer.

**How many devices can one account hold?**
One VeltronHub X1 and up to 64 VeltronSense sensors per hub. Additional hubs require
separate accounts.

**Does Veltron sell to businesses with invoicing?**
Yes, business accounts receive monthly PDF invoices with purchase-order support. Contact
the business desk; a business account is a different account type and cannot be converted
from a consumer account by support.

**Does the system work during a power cut?**
Automations that do not require internet access continue to run. Remote access, voice
assistants and notifications stop until power is restored.

**What firmware version am I on?**
Settings > Device > Firmware > Current version. Quote the exact version in any ticket.

**Can support reset my password?**
No. Support cannot read or set a password. Use the password-reset link sent by email.

**Is there an API for developers?**
Yes, the Veltron Local API on the hub, documented in the developer documentation, and a
read-only cloud API for Pro plans. The local API requires a hub on the same network and
needs no account.
        """,
    },
    {
        "doc_id": "faq-privacy-serbian",
        "category": "faq",
        "title": "Česta pitanja (srpski)",
        "language": "sr",
        "tags": ["faq", "sr", "srpski", "porudžbina", "reklamacija"],
        "body": """
# Česta pitanja

**Da li uređaj radi bez interneta?**
Da. Lokalno upravljanje, automatizacije i senzori rade na lokalnoj mreži. Udaljeni
pristup i istorija duža od 7 dana zahtevaju Veltron Cloud pretplatu.

**Koliko uređaja može jedan nalog da drži?**
Jedan VeltronHub X1 i najviše 64 VeltronSense senzora po hubu. Dodatni hubovi zahtevaju
zasebne naloge.

**Da li korisnička podrška može da promeni moju lozinku?**
Ne. Podrška ne može da čita niti postavlja lozinku. Koristite link za promenu lozinke
poslat na vašu adresu e-pošte.

**Kada dobijam račun?**
Račun se izdaje automatski na dan obračuna i šalje se na adresu za naplatu na nalogu.
Ponovno izdavanje računa traje 2 radna dana.

**Kako da vratim proizvod?**
Fizički proizvodi mogu se vratiti u roku od 30 dana od isporuke, u originalnom pakovanju
i sa svim priborom. Troškove povratne pošiljke snosi kupac.

**Šta radim ako uređaj ne radi?**
Prvo pratite korak po korak uputstvo za rešavanje problema za vaš uređaj. Ako problem
i dalje postoji, otvorite tiket sa serijskim brojem uređaja i tačnim vremenima pojavljivanja
greške.

**Kako da kontaktiram podršku?**
Otvorite tiket u aplikaciji Veltron ili na stranici za podršku. Za hitne problemente
koristite oznaku "hitno". Podrška odgovara u roku od 1 radnog dana.
        """,
    },
]

ALL_DOCS: list[dict[str, Any]] = PRODUCTS + POLICIES + TROUBLESHOOTING + FAQ


def kb_documents() -> list[dict[str, Any]]:
    """Every knowledge-base document, stamped with the synthetic-data disclaimer."""
    out = []
    for d in ALL_DOCS:
        rec = dict(d)
        rec["synthetic"] = True
        rec["disclaimer"] = DEMO_DISCLAIMER
        rec["text"] = f"{DEMO_DISCLAIMER}\n\n{d['body'].strip()}\n"
        out.append(rec)
    return out


def load_knowledge_base(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the KB, optionally overlaying JSON/Markdown files from ``knowledge_base/``.

    The bundled :data:`ALL_DOCS` list is always the base so the demo works from a clean
    checkout; files on disk can add or replace documents without code changes.
    """
    docs = {d["doc_id"]: d for d in kb_documents()}
    root = Path(root) if root else KB_ROOT
    if root.exists():
        for path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                log.error("skipping malformed KB file %s: %s", path, exc)
                continue
            for item in payload if isinstance(payload, list) else payload.get("documents", []):
                item = dict(item)
                item.setdefault("synthetic", True)
                item["disclaimer"] = DEMO_DISCLAIMER
                item["text"] = f"{DEMO_DISCLAIMER}\n\n{item.get('body', '').strip()}\n"
                docs[item["doc_id"]] = item
        log.info("knowledge base: %d bundled documents, %d total after overlay",
                 len(kb_documents()), len(docs))
    return list(docs.values())


def get_doc(doc_id: str) -> dict[str, Any] | None:
    for d in load_knowledge_base():
        if d["doc_id"] == doc_id:
            return d
    return None


__all__ = [
    "PRODUCTS",
    "POLICIES",
    "TROUBLESHOOTING",
    "FAQ",
    "ALL_DOCS",
    "DEMO_DISCLAIMER",
    "kb_documents",
    "load_knowledge_base",
    "get_doc",
]
