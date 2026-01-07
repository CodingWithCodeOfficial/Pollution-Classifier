from django import template

register = template.Library()

@register.filter
def replace_underscore(value):
    """Replace underscores with spaces in a string."""
    if not value:
        return value
    return str(value).replace('_', ' ')
