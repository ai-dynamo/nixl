#ifndef _NIXL_LIST_ELEM_H
#define _NIXL_LIST_ELEM_H


template <typename T>
class nixlLinkElem {
private:
    T *_next;
public:
    nixlLinkElem()
    {
        _next = NULL;
    }

    ~nixlLinkElem() {
        _next = NULL;;
    }

    /* Link this element into the chain afer "elem" */
    void link(T *elem)
    {
        elem->_next = _next;
        _next = elem;
    }

    /* Exclude this element from the chain, return the new head */
    T *unlink()
    {
        T *ret = _next;
        /* Forget my place */
        _next = NULL;
        return ret;
    }

    T *next()
    {
        return _next;
    }

} ;

#endif