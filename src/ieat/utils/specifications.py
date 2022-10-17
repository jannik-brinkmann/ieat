from collections import namedtuple, OrderedDict

AssociationTestSpecification = namedtuple('AssociationTestSpecification', ['X', 'Y', 'A', 'B'])

ASSOCIATION_TESTS = OrderedDict(
    [
        (
            'Gender-Science',
            AssociationTestSpecification('gender/male', 'gender/female', 'gender/science', 'gender/liberal-arts')
        ),
        (
            'Gender-Career',
            AssociationTestSpecification('gender/male', 'gender/female', 'gender/career', 'gender/family')
        )
    ]
)
