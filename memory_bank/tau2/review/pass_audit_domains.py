#!/usr/bin/env python3
"""Domain-specific halves of pass_audit.py, one module per tau2 domain.

pass_audit.py keeps everything that holds in any tau2 domain — mixed turns, tool errors, writes
without confirmation, fake transfers, claimed-but-not-performed actions, and the rule that a
mechanical finding outranks the model. What cannot be shared is the one check that matters most
on a PASSING trajectory: whether what the agent TOLD the customer matches what the tools actually
returned. The fields that carry that (membership and cabin in airline, order status and item
availability in retail, line id and roaming state in telecom) have nothing in common, so each
domain supplies its own.

Each module exposes:
    WRITE_CLAIM_VERBS  regex alternation of verbs whose past tense implies a write happened
    extract_facts(messages)          -> dict of what the tools actually said
    claim_checks(messages, facts, user_tools) -> findings anchored to assistant turns
    context_lines(facts)             -> lines for the <Pass Context> block
    prompt(): dict of the four slots the shared SYSTEM prompt parameterizes

A finding carries check= and detail=. Checks listed in HARD_FACTS (in pass_audit.py) are true by
construction and the model may not overrule them; everything else is a lead it must verify.
"""
import json, re


def _tool_json(m):
    try:
        return json.loads(m.get("content") or "")
    except Exception:
        return None


def _agent_turns(messages):
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and (m.get("content") or "").strip():
            yield i, m["content"], m["content"].lower()


# =============================================================================
# airline
# =============================================================================
class airline:
    WRITE_CLAIM_VERBS = r"(cancel|refund|book|updat|modif|chang|issu|sent|added|remov)"

    @staticmethod
    def extract_facts(messages):
        memberships, insurance, cabins = set(), set(), set()
        fetched_user = fetched_res = False
        for m in messages:
            if m["role"] != "tool":
                continue
            c = m.get("content") or ""
            blob = _tool_json(m)
            if isinstance(blob, dict):
                if "membership" in blob:
                    fetched_user = True; memberships.add(str(blob["membership"]).lower())
                if "insurance" in blob and "reservation_id" in blob:
                    fetched_res = True; insurance.add(str(blob["insurance"]).lower())
                if "cabin" in blob and "reservation_id" in blob:
                    cabins.add(str(blob["cabin"]).lower())
            else:
                for mm in re.finditer(r'"membership":\s*"(\w+)"', c):
                    fetched_user = True; memberships.add(mm.group(1).lower())
                for mm in re.finditer(r'"insurance":\s*"(\w+)"', c):
                    fetched_res = True; insurance.add(mm.group(1).lower())
        return dict(memberships=sorted(memberships), insurance=sorted(insurance),
                    cabins=sorted(cabins), fetched_user=fetched_user, fetched_res=fetched_res)

    @staticmethod
    def claim_checks(messages, facts, user_tools=()):
        out = []
        memberships, insurance, cabins = set(facts["memberships"]), set(facts["insurance"]), set(facts["cabins"])
        fetched_user, fetched_res = facts["fetched_user"], facts["fetched_res"]
        for i, text, low in _agent_turns(messages):
            claimed = set()
            for mm in re.finditer(r"(you are (a|an) |you're (a|an) |as (a|an) |your membership( level)?( is| status is)?:? (is )?)(gold|silver|regular)\b", low):
                ctx = low[max(0, mm.start() - 40):mm.start()]
                if re.search(r"\b(if|unless|whether|only|when|for|since you're not|are not)\b\s*$", ctx) or "/" in low[mm.start():mm.end() + 8]:
                    continue
                claimed.add(mm.group(mm.lastindex))
            for mm in re.finditer(r"\b(gold|silver|regular) member, you\b", low):
                if not re.search(r"\b(if|as long as|when)\b\s*(a|an)?\s*$", low[max(0, mm.start() - 20):mm.start()]):
                    claimed.add(mm.group(1))
            for lvl in sorted(claimed):
                if fetched_user and lvl not in memberships:
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent tells the customer they are a {lvl} member; get_user_details returned {sorted(memberships)}"))
                elif not fetched_user:
                    out.append(dict(turn_idx=i, check="claim_without_lookup",
                                    detail=f"agent tells the customer they are a {lvl} member without ever calling get_user_details"))
            if fetched_res:
                if re.search(r"\b(you have|you do have|your reservation (has|includes)|you purchased|shows? (travel )?insurance: ?yes)\b", low) and not re.search(r"\b(if|unless|whether|no|not|without|don.t|doesn.t|does not|do not)\b[^.]{0,30}insurance", low) and "yes" not in insurance:
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent says the customer has travel insurance; reservations returned insurance={sorted(insurance)}"))
                if re.search(r"\b(no|not|without|don.t have|does not have|do not have|not purchased?)\b[^.]{0,30}(travel )?insurance", low) and "no" not in insurance and insurance:
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent says the customer has NO travel insurance; reservations returned insurance={sorted(insurance)}"))
        cancelled = any(m["role"] == "tool" and re.search(r'"status":\s*"cancelled"', (m.get("content") or "")) for m in messages)
        for i, text, low in _agent_turns(messages):
            if re.search(r"(you (have|had) a business[- ]class (flight|reservation)|your (flight|reservation) (is|was) (in )?business[- ]class|is part of a business[- ]class reservation)", low) and cabins and "business" not in cabins:
                out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                detail=f"agent says the customer's flight is business class; reservations returned cabin={sorted(cabins)}"))
            if re.search(r"(your (flight|business flight) (that )?(was|has been|got) cancel+ed|i see that you have a .{0,30}flight that was cancel+ed)", low) and not cancelled:
                out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                detail="agent says the customer's flight was cancelled; no tool result shows a cancelled flight status"))
            if re.search(r"(i can offer (you )?(a|an) (\$?\d+ )?(travel )?certificate|as a gesture[^.]{0,60}certificate|offer (you )?a (\$?\d+ )?(travel )?(certificate|voucher) as a gesture|gesture of (goodwill|sympathy|support)[^.]{0,60}(certificate|voucher))", low):
                out.append(dict(turn_idx=i, check="compensation_offered",
                                detail=f"agent offers a certificate/voucher here — verify the policy allows it (only for delayed/cancelled flights, silver/gold or insured or business, and only if the customer asked): {text[:100]!r}"))
        return out

    @staticmethod
    def context_lines(facts):
        return [f"Records seen in tool results: membership={facts['memberships'] or 'not fetched'}, "
                f"insurance={facts['insurance'] or 'not fetched'}, cabin={facts['cabins'] or 'n/a'}."]

    @staticmethod
    def prompt():
        return dict(
            false_statement_eg='calls a regular member "Gold", says a refund is possible when policy forbids it, quotes a wrong price',
            compensation_line="proactive_compensation: offered a certificate/refund/voucher the customer had not asked for or that policy does not allow.",
            customer_lies_eg="claims Gold, claims insurance, invents a cancelled flight",
            domain_checks="`claim_contradicts_record`, `claim_without_lookup` and `compensation_offered` are regex leads and may be false; verify them against the quoted turn.")


# =============================================================================
# retail
# =============================================================================
class retail:
    WRITE_CLAIM_VERBS = r"(cancel|refund|return|exchang|book|updat|modif|chang|issu|sent|added|remov|process)"

    @staticmethod
    def extract_facts(messages):
        """Order status, item availability and the payment methods on file — the three things a
        retail agent most often states to the customer and most often gets wrong."""
        orders, avail, prices, pay = {}, {}, {}, {}
        for m in messages:
            if m["role"] != "tool":
                continue
            blob = _tool_json(m)
            if not isinstance(blob, dict):
                continue
            if "order_id" in blob and "status" in blob:
                orders[blob["order_id"]] = blob["status"]
                for it in blob.get("items") or []:
                    if it.get("item_id"):
                        prices[it["item_id"]] = it.get("price")
            if "variants" in blob:
                for v in (blob["variants"] or {}).values():
                    if v.get("item_id"):
                        avail[v["item_id"]] = bool(v.get("available"))
                        prices.setdefault(v["item_id"], v.get("price"))
            if "payment_methods" in blob:
                for k, v in (blob["payment_methods"] or {}).items():
                    pay[k] = v
        return dict(order_status=orders, availability=avail, prices=prices, payment_methods=pay)

    @staticmethod
    def claim_checks(messages, facts, user_tools=()):
        out = []
        st = facts["order_status"]; avail = facts["availability"]; pay = facts["payment_methods"]
        brands = {(v.get("brand") or "").lower(): k for k, v in pay.items() if v.get("brand")}
        lastfour = {v.get("last_four"): k for k, v in pay.items() if v.get("last_four")}
        for i, text, low in _agent_turns(messages):
            # 1. order status stated to the customer vs the record
            for mm in re.finditer(r"(#W\d{7})[^.]{0,80}?\b(is|status is|currently)\b[^.]{0,30}?\b(pending|delivered|cancelled|canceled|processed)\b", low):
                oid = mm.group(1).upper(); said = mm.group(3).replace("canceled", "cancelled")
                actual = st.get(oid)
                if actual and said != str(actual).lower():
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent tells the customer order {oid} is '{said}'; get_order_details returned status='{actual}'"))
            # 2. the retail rule the agent most often states backwards: pending is exactly when
            #    a cancellation IS allowed, so "cannot cancel because it is pending" is inverted.
            if re.search(r"\b(can'?t|cannot|unable to|not able to|isn'?t possible to)\b[^.]{0,60}\bcancel\b", low) and \
               re.search(r"\bbecause\b[^.]{0,60}\b(pending|still pending|in pending status)\b", low):
                out.append(dict(turn_idx=i, check="policy_stated_backwards",
                                detail=f"agent says the order cannot be cancelled BECAUSE it is pending; retail policy allows cancelling exactly while an order is pending: {text[:120]!r}"))
            # 3. availability stated vs get_product_details
            for mm in re.finditer(r"\b(\d{10})\b[^.]{0,60}?\b(is|are)\b[^.]{0,20}?\b(not available|unavailable|out of stock|available|in stock)\b", low):
                iid = mm.group(1); said_avail = mm.group(3) in ("available", "in stock")
                if iid in avail and avail[iid] != said_avail:
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent says item {iid} is {'available' if said_avail else 'not available'}; get_product_details returned available={avail[iid]}"))
            # 4. a card the customer does not have on file. Only an ASSERTION that it is the
            #    customer's card counts ("your Amex", or a brand with a last-four); sentences that
            #    deny, refuse or offer it ("I cannot refund to Amex", "in the future?") are skipped.
            for mm in re.finditer(r"\byour\s+(\w+\s+){0,2}(visa|mastercard|amex|american express)\b[^.]{0,40}?(ending (in )?(\d{4}))?"
                                  r"|\b(visa|mastercard|amex|american express)\b\s*(card\s*)?ending (in )?(\d{4})", low):
                sent = low[max(0, low.rfind('.', 0, mm.start()) + 1): low.find('.', mm.end()) + 1 or None]
                if re.search(r"\b(not available|not on file|cannot|can'?t|don'?t have|does not have|do not have|no longer|isn'?t|is not|unable to)\b", sent) \
                        or re.search(r"\b(would you like|do you want|shall i|can i|in the future|if you)\b", sent):
                    continue
                g = mm.group(0)
                bm = re.search(r"visa|mastercard|amex|american express", g)
                brand = bm.group(0).replace("american express", "amex")
                if pay and brand not in brands:
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent names a {brand} card; the payment methods on file are {[(v.get('source'), v.get('brand'), v.get('last_four')) for v in pay.values()]}"))
                    break
                fm = re.search(r"ending (?:in )?(\d{4})", g)
                four = fm.group(1) if fm else None
                if four and lastfour and four not in lastfour:
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent says the card ends in {four}; the cards on file end in {sorted(x for x in lastfour if x)}"))
                    break
            # 5. a refund figure, for the model to check against the item prices it saw
            mm = re.search(r"refund (of |amount(ing)? (of |to )?)?\$([\d,]+\.\d{2})", low)
            if mm:
                out.append(dict(turn_idx=i, check="amount_stated",
                                detail=f"agent states a refund of ${mm.group(4)} — verify it against the item prices in the tool results (a wrong total is a wrong_calculation, not a false_statement)"))
        return out

    @staticmethod
    def context_lines(facts):
        st = facts["order_status"]
        return [f"Order statuses seen in tool results: {st or 'none fetched'}.",
                f"Payment methods on file: {[(v.get('source'), v.get('brand'), v.get('last_four')) for v in facts['payment_methods'].values()] or 'not fetched'}.",
                f"Item availability seen: {sum(1 for v in facts['availability'].values() if v)} available / {len(facts['availability'])} variants fetched."]

    @staticmethod
    def prompt():
        return dict(
            false_statement_eg='tells the customer an order is "delivered" when the record says "pending", says an item is unavailable when get_product_details shows available=true, names a card the customer does not have on file, quotes a refund total that does not follow from the item prices',
            compensation_line="proactive_compensation: offered a refund, a discount, free shipping or an exception the customer had not asked for or that policy does not allow.",
            customer_lies_eg="misremembers an order id, claims an item was damaged, asks to refund to a different payment method than the one used",
            domain_checks="`claim_contradicts_record`, `policy_stated_backwards` and `amount_stated` are regex leads and may be false; verify them against the quoted turn. For `amount_stated` you must do the arithmetic yourself from the item prices in the tool results.")


# =============================================================================
# telecom
# =============================================================================
class telecom:
    WRITE_CLAIM_VERBS = r"(suspend|resume|enabl|disabl|refuel|activat|restor|updat|chang|process|appli|sent|added|remov)"
    LINE_WRITES = ("enable_roaming", "disable_roaming", "suspend_line", "resume_line", "refuel_data", "send_payment_request")

    @staticmethod
    def extract_facts(messages):
        """Which line belongs to which phone number, and the per-line state the agent narrates.
        The customer's own number comes from get_customer_by_* ; a line's number comes from
        get_details_by_id. When the agent writes to a line whose number is not the customer's,
        that is a fact, not a guess."""
        lines, customer_phone, customer_id, bills = {}, None, None, {}
        for m in messages:
            if m["role"] != "tool":
                continue
            blob = _tool_json(m)
            if not isinstance(blob, dict):
                continue
            if "line_id" in blob:
                lines.setdefault(blob["line_id"], {}).update(
                    {k: blob[k] for k in ("phone_number", "status", "roaming_enabled",
                                          "data_used_gb", "data_limit_gb", "data_refueling_gb", "plan_id") if k in blob})
            if "customer_id" in blob and "full_name" in blob:
                customer_id = blob["customer_id"]
                customer_phone = blob.get("phone_number") or customer_phone
            for b in (blob.get("bills") or []):
                if isinstance(b, dict) and b.get("bill_id"):
                    bills[b["bill_id"]] = b.get("total_due")
        return dict(lines=lines, customer_phone=customer_phone, customer_id=customer_id, bills=bills)

    @staticmethod
    def claim_checks(messages, facts, user_tools=()):
        out = []
        lines = facts["lines"]; phone = facts["customer_phone"]
        device_tools = set(user_tools or ())
        for i, m in enumerate(messages):
            if m["role"] != "assistant":
                continue
            # HARD: a write aimed at a line that is not the customer's own number
            for t in m.get("tool_calls") or []:
                if t["name"] in telecom.LINE_WRITES:
                    args = t.get("arguments") or {}
                    lid = args.get("line_id")
                    known = (lines.get(lid) or {}).get("phone_number")
                    if phone and lid and known and str(known) != str(phone):
                        out.append(dict(turn_idx=i, check="wrote_to_wrong_line",
                                        detail=f"{t['name']} was called on {lid}, whose phone_number is {known}, but the customer's number is {phone}"))
                    elif phone and lid and lid not in lines:
                        out.append(dict(turn_idx=i, check="wrote_to_unverified_line",
                                        detail=f"{t['name']} was called on {lid} without ever fetching that line's details, so its phone number was never checked against the customer's {phone}"))
            # HARD: the agent tried to operate the customer's handset itself
            if i + 1 < len(messages) and messages[i + 1]["role"] == "tool":
                c = messages[i + 1].get("content") or ""
                mm = re.search(r"Tool '([\w_]+)' not found", c)
                if mm and mm.group(1) in device_tools:
                    out.append(dict(turn_idx=i, check="called_customer_device_tool",
                                    detail=f"the agent called {mm.group(1)}, which runs on the customer's phone; it must ask the customer to do it instead"))
        for i, text, low in _agent_turns(messages):
            # roaming state narrated to the customer vs the line record
            # "is not roaming enabled" must not read as "enabled": alternation is first-match, so
            # the negative forms have to come first, and a bare "not"/"isn't" in between counts too.
            for mm in re.finditer(r"\b(l\d{4})\b([^.]{0,70}?)\broaming\b([^.]{0,30}?)\b(is |has been |now )?"
                                  r"(not enabled|not turned on|disabled|turned off|enabled|turned on)\b", low):
                lid = mm.group(1).upper()
                negated = re.search(r"\b(not|isn'?t|aren'?t|no longer|never)\b", mm.group(2) + mm.group(3))
                said_on = mm.group(5) in ("enabled", "turned on") and not negated
                rec = (lines.get(lid) or {}).get("roaming_enabled")
                if rec is not None and bool(rec) != said_on:
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent says roaming on {lid} is {'enabled' if said_on else 'disabled'}; the line record says roaming_enabled={rec}"))
            # line status narrated vs record
            for mm in re.finditer(r"\b(l\d{4})\b[^.]{0,60}?\b(is |status is )?(active|suspended)\b", low):
                lid = mm.group(1).upper(); said = mm.group(3)
                rec = (lines.get(lid) or {}).get("status")
                if rec and str(rec).lower() != said:
                    out.append(dict(turn_idx=i, check="claim_contradicts_record",
                                    detail=f"agent says {lid} is {said}; the line record says status='{rec}'"))
            # a data figure, for the model to check
            if re.search(r"\b\d+(\.\d+)?\s?gb\b", low) and lines:
                out.append(dict(turn_idx=i, check="amount_stated",
                                detail=f"agent states a data figure — verify against data_used_gb / data_limit_gb in the tool results: {text[:110]!r}"))
        return out

    @staticmethod
    def context_lines(facts):
        ls = {k: {kk: vv for kk, vv in v.items() if kk in ("phone_number", "status", "roaming_enabled", "data_used_gb")}
              for k, v in facts["lines"].items()}
        return [f"Customer: id={facts['customer_id']}, phone={facts['customer_phone'] or 'never fetched'}.",
                f"Lines seen in tool results: {ls or 'none'}.",
                f"Bills seen: {facts['bills'] or 'none'}."]

    @staticmethod
    def prompt():
        return dict(
            false_statement_eg='tells the customer roaming is enabled on a line whose record says otherwise, reports a data allowance that does not match get_data_usage, says a line is active when it is suspended',
            compensation_line="proactive_compensation: offered a credit, a data refuel, a plan change or a fee waiver the customer had not asked for or that policy does not allow.",
            customer_lies_eg="gives a line id that is not theirs, claims a setting is already on, misreports what their phone shows",
            domain_checks=("`wrote_to_wrong_line`, `wrote_to_unverified_line` and `called_customer_device_tool` are computed "
                           "from the transcript and are TRUE by construction — a write aimed at the wrong line, or the agent "
                           "trying to operate the handset itself, is critical even when the task still passed. "
                           "`claim_contradicts_record` and `amount_stated` are regex leads; verify them against the quoted turn."))


BY_DOMAIN = {"airline": airline, "retail": retail, "telecom": telecom}

# Findings a domain module computes from the transcript rather than guesses at: the model judges
# their severity and consequence but may not claim they did not happen.
DOMAIN_HARD_FACTS = {"wrote_to_wrong_line", "wrote_to_unverified_line", "called_customer_device_tool"}

# Of those, the ones that on their own make a pass fake. Acting on someone else's record is always
# critical: the task can reach its target state while another customer's line was changed. Calling
# a handset tool is NOT here: it returns "Tool not found" and the agent typically then asks the
# customer to do it, a recovered slip like a corrected tool_error, which the prompt calls minor.
DOMAIN_CRITICAL_FACTS = {"wrote_to_wrong_line"}
