"""V2.1 budget order only; shared V2 renderer owns all integrity and sizing."""
from .ranking_v2 import order

OBJECT_KINDS=frozenset(('indexed_object','directory_object'))

def actionable(f):
    v=f.value
    return f.kind in OBJECT_KINDS and isinstance(v.get('path'),str) and isinstance(v.get('file_id'),str) and bool(v['file_id'])

def select(*,new,old,parent,validated,initial,free,limit,ctx,settings,add,summary,fair_fill):
    def cap(fraction):return initial+int(free*fraction)
    schema_new=any(validated[n].shape=='sqlite_schema' for n in new)
    def one(nids,predicate,budget,tier):
        for nid in nids:
            for f in order([f for f in validated[nid].facts if predicate(f)],ctx,settings):
                if add(nid,f,budget,tier):break
    # NEW always receives the first opportunity; KEPT navigation is considered
    # before the NEW result can spend the entire detail budget.
    for nid in new:summary(nid,cap(.12))
    one(new,actionable,cap(.28),'NEW_ACTIONABLE')
    mixed=[]
    for i in range(max(len(new),len(old),0)):
        if i<len(new):mixed.append(new[i])
        if i<len(old):mixed.append(old[i])
    fair_fill(new,lambda f:f.kind in ('table_directory','ordered_columns'),cap(.55 if schema_new else .45),'NEW_TABLE_DIRECTORY')
    # Interleave name slices across NEW and KEPT, bounded independently from
    # executable bindings. Long result catalogs cannot take all detail space.
    fair_fill(mixed,lambda f:f.kind=='object_names',cap(.52),'NAVIGATION_DIRECTORY')
    fair_fill(old,lambda f:f.kind in ('table_directory','ordered_columns'),cap(.55),'KEPT_TABLE_DIRECTORY')
    # Column names and observed typed fields have separate opportunities. Table
    # catalogs cannot consume the entire schema allocation; columns remain bound
    # to their original table and exact source locators.
    fair_fill(new+old,lambda f:f.kind=='column_directory',cap(.78 if schema_new else .69),'COLUMN_DIRECTORY')
    fair_fill(new+old,lambda f:f.kind=='schema_field',cap(.92 if schema_new else .78),'OBSERVED_SCHEMA_FIELD')
    one(old,actionable,cap(.82),'KEPT_ACTIONABLE')
    fair_fill(new,actionable,cap(.94),'NEW_OBJECT_DETAILS')
    fair_fill(old,actionable,cap(.96),'KEPT_OBJECT_DETAILS')
    for nid in old:summary(nid,cap(.95))
    fair_fill(new+old,lambda f:f.level!='L3' and f.kind not in ('object_names','table_directory','path_distribution'),cap(.96),'OTHER_DETAILS')
    for nid in parent:summary(nid,cap(.97))
    fair_fill(parent,lambda f:f.level!='L3',cap(.98),'PARENT_DETAILS')
    # Reuse space for navigation before literals/operation text, including old
    # directories. All caps remain optional; identity/edges never depend on them.
    fair_fill(mixed,lambda f:f.kind in OBJECT_KINDS or f.kind in ('object_names','table_directory','column_directory','schema_field'),limit,'NAVIGATION_SLACK')
    fair_fill(new+old+parent,lambda f:f.level!='L3',limit,'OTHER_SLACK')
